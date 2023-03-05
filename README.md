# Bitonic Sort
Bitonic sorting written with OpenCL. Works an order of magnitude faster than std::sort().
Task: input number N, then N numbers. Output a sorted array.

## Requirements 

The following applications have to be installed:

1. CMake 3.2 version (or higher)
2. GTest
3. Compiliter c++17
4. OpenCL v2.2


## Compiling 

To compile each of the cache you need to use сmake in the directory build:

``` cmd
$ git submodule init && git submodule update
$ mkdir build
$ cd build
$ сmake ..
$ cmake --build
```

To select the type of input surrenders or platform use:

``` cmd
$ сmake .. -DTYPE=type -DPLATFORM=platform
$ cmake --build .
```
where "type" is the built-in type of the C/C++ languages (except char), "platform" is NVIDIA, INTEL or ANY_PLATFORM. The default setting is int and NVIDIA.
## Run the program:

You can find all binaries in dir build/bin


Bitonic Sort: 
``` cmd
$ ./bin/bsort
```
Test for Sort:

``` cmd
$ ./bin/testBsort
```
