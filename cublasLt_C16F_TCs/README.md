# cublasLt_C16F_TCs

Matrix multiplication of complex, half-precision using Tensor Cores

There are two scenarios that use tensor operation with complex half precision.
* CUDA_C_16F, CUDA_C_16F, CUDA_C_16F, CUDA_C_32F, CUDA_C_32F
* CUDA_C_16F, CUDA_C_16F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F

First, set IDENTITY 1 and PRINT 1. This will create 2 input identity matrices, in matrix A and B. The result should print a 16x16 identity matrix. You'll notice they are pairs, to show real and imaginary parts.

Next, set PRINT 0 and you can test multiple square matrices with the identity test.

Lastly, set IDENTITY 0 and PRINT 0 and you will test multiple square and non-square matrices with randomly generated data.

You should notice another define variable TIME_TRANSFORM. When matrix A and B are generated with random numbers, they were generated with an interleaved layout. Meaning data is stored [real, imaginary, real, imaginary, ...]. In order to utilize Tensor Cores the data must be in planar layout. Meaning data is stored [real, real, real, .... (half way), imaginary, imaginary, imaginary].

When TIME_TRANSFORM 1, then the time taken to transform A and B to planar layout, perform matrix multiplication, and the time taken to transform C from planar to interleaved layout is calculated. When TIME_TRANSFORM 0 only matrix multiplication is profiled.

**This example requires compute capability 7.0 and greater.**

## Getting started
Import example into Eclipse with NsightEE plugin and build.

### Prerequisites
CUDA toolkit with symbolic link to the CUDA directory ```/usr/local/cuda.```.

## Built With
These examples utilize the following toolsets:
* cuBLASLt
* Thrust
* C++11
