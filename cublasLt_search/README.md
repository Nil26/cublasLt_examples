Exhaustive search of algorithms

This example demonstrates how to use the cuBLASLt library to do an exhaustive search to obtain valid algorithms for GEMM.

There are four scenarios performing half and single precision GEMMS.

CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F
CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F
CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F
CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F

There is one scenario performing a complex, single precision GEMM.

CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F

First, set IDENTITY 1 and PRINT 1. This will create 2 input identity matrices, in matrix A and B. The result should print a 16x16 identity matrix. You'll notice they are pairs, to show real and imaginary parts.

Next, set PRINT 0 and you can test multiple square matrices with the identity test.

Lastly, set IDENTITY 0 and PRINT 0 and you will test multiple square and non-square matrices with randomly generated data.

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
