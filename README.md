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
$ mkdir build
$ cd build
$ сmake ..
$ make
```

To select the type of input surrenders use:

``` cmd
$ сmake .. -DTYPE=type
$ make
```
where "type" is the built-in type of the C/C++ languages (except char). The default setting is int.
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
