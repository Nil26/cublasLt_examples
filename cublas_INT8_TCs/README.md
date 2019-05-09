# cublasLt_INT8_TCs

Matrix multiplication of IGEMM

This example demonstrates how to use the cuBLASLt library to perform cublasLtMatmul with tensor operations IGEMM, with the memory order transforms on all buffers.

For better performance the data order transforms should be offline as much as possible.

Transa, transb assumed m = n = k; alpha, beta are host pointers;
For transforms, alpha assumed 1, beta assumed 0, and stream assumed 0.

**This example requires compute capability 7.2 and greater.**

## Getting started
Import example into Eclipse with NsightEE plugin and build.

### Prerequisites
CUDA toolkit with symbolic link to the CUDA directory ```/usr/local/cuda.```.

## Built With
These examples utilize the following toolsets:
* cuBLASLt
* Thrust
* C++11
