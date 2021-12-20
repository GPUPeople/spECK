# Disclaimer
This repository is based on *spECK: Accelerating GPU Sparse Matrix-Matrix Multiplication Through Lightweight Analysis* ([ACM Link](https://dl.acm.org/doi/10.1145/3332466.3374521)). Please cite this work if you use it for research.

# Getting Started
1. Install CUDA 10.1, 10.2 or 11.0 from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. If you use CUDA 11.0 or newer, please switch to _cuda11_ branch
3. Install g++ > 7.0 and gcc on Linux or Visual Studio with "Desktop development with C++" workload
4. Install CMake 3.15.5 or newer from [https://cmake.org/](https://cmake.org/)
5. Set spECK_STATIC_MEM_PER_BLOCK and spECK_DYNAMIC_MEM_PER_BLOCK in include/Multiply.h line 9 & 10 to the values your hardware supports
    - spECK_STATIC_MEM_PER_BLOCK should be 49152 for all recent devices
    - spECK_DYNAMIC_MEM_PER_BLOCK should be 
        - 49152 for all devices before Volta and Turing (cc < 7.0)
        - 65536 for Turing devices (cc7.5)
        - 98304 for Volta devices (cc7.0)
        - 101376 for Ampere consumer devices (RTX 30xx) (cc8.6)
        - 166912 for Ampere professional devices (e.g. A100) (cc8.0)
    - if you do not know your GPU generation or hardware limits, compile and run spECK and it will throw errors with information about the correct values
6. Build
    - Windows (use CMake GUI to setup project) or
        o Build the project using "cmake -DCUDA_BUILD_CC86=TRUE -S . -B build -A x64" (Set CCXX to correct Compute Capability) followed by opening "runSpeck.sln", select "Release" configuration and build
        o run spECK using ".\build\Release\runspECK.exe <path-to-csr-matrix> config.ini"
    - Linux
        o Set the correct ComputeCapability (Default is CC70) in "linuxsetup.sh" and run "./linuxsetup.sh"
        o run spECK using "./build/runspECK <path-to-csr-matrix> config.ini"

# Notes

spECK is compiled into a library "spECKLib.lib" and an executable "runspECK.exe" (or linux equivalents).
The executable comes with an .mtx (Matrix Market File Format) reader which converts the .mtx file into and saves an ".hicsr" binary file for faster runtimes.

runspECK takes 2 input parameters:
- path to matrix in .mtx file format (required)
- path to config.ini (optional)

Config.ini contains helpful options for:
- TrackCompleteTimes: enable/disable benchmarking
- TrackIndividualTimes: enable/disable benchmarking of all stages of speck (comes with performance overhead)
- CompareResult: enable/disable result matrix structure comparison with CUSPARSE. Prints an error message if column indices do not match
- IterationsWarmUp/IterationsExecution: set number of warm up and execution iteration for benchmarking. WarmUp is helpful to make sure that the GPU is running at it's highest clock rate
- InputFile: override input matrix - if an input file is defined in the config, this will override the first command line parameter


# Bibtex
```
@inproceedings{10.1145/3332466.3374521,
author = {Parger, Mathias and Winter, Martin and Mlakar, Daniel and Steinberger, Markus},
title = {SpECK: Accelerating GPU Sparse Matrix-Matrix Multiplication through Lightweight Analysis},
year = {2020},
isbn = {9781450368186},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3332466.3374521},
doi = {10.1145/3332466.3374521},
booktitle = {Proceedings of the 25th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming},
pages = {362–375},
numpages = {14},
location = {San Diego, California},
series = {PPoPP ’20}
}
``` 
