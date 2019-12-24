# Getting Started
CUDA toolkit with symbolic link to the CUDA directory ```/usr/local/cuda.```.

## Built With
These examples utilize the following toolsets:
* cuBLASLt
* Thrust
* C++11

## cublasLt_sgemm

Matrix multiplication of SGEMM

This example demonstrates how to use the cuBLASLt library to perform SGEMM. It is nearly a drop-in replacement for cublasSgemm. It performs multiplications on input/output/compute types CUDA_R_32F.

It cycles through square matrices up to _maxN_. Computes the reference matrix, then performs matrix multiplication on the device, downloads the output matrix, and compares the answer.

When ROW_MAJOR 1, the host reference output and cuBLASLt is calculate with row-major format. Otherwise, column-major format.

Note: This example computes a reference answer on the host side and can take awhile to process in N is large.

**This example has been tested with compute capability 6.0 and greater.**

## cublasLt_search

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

## cublasLt_C16F_TCs

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

## cublasLt_INT8_TCs

Matrix multiplication of IGEMM

This example demonstrates how to use the cuBLASLt library to perform cublasLtMatmul with tensor operations IGEMM, with the memory order transforms on all buffers.

For better performance the data order transforms should be offline as much as possible.

Transa, transb assumed m = n = k; alpha, beta are host pointers;
For transforms, alpha assumed 1, beta assumed 0, and stream assumed 0.

**This example requires compute capability 7.2 and greater.**
