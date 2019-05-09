# cublasLt_sgemm

Matrix multiplication of SGEMM

This example demonstrates how to use the cuBLASLt library to perform SGEMM. It is nearly a drop-in replacement for cublasSgemm. It performs multiplications on input/output/compute types CUDA_R_32F.

It cycles through square matrices up to _maxN_. Computes the reference matrix, then performs matrix multiplication on the device, downloads the output matrix, and compares the answer.

When ROW_MAJOR 1, the host reference output and cuBLASLt is calculate with row-major format. Otherwise, column-major format.

Note: This example computes a reference answer on the host side and can take awhile to process in N is large.

**This example has been tested with compute capability 6.0 and greater.**


## Getting started
Import example into Eclipse with NsightEE plugin and build.

### Prerequisites
CUDA toolkit with symbolic link to the CUDA directory ```/usr/local/cuda.```.

## Built With
These examples utilize the following toolsets:
* cuBLASLt
* Thrust
* C++11
