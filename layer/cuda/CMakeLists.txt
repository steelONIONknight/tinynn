cmake_minimum_required(VERSION 3.10)
project(CudaOperator CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_library(CudaOperator
        absval_cuda.h absval_cuda.cpp absval_cuda.cu
        test_cuda.cu test_cuda.h
        innerproduct_cuda.cpp innerproduct_cuda.h innerproduct_cuda.cu
        convolution_cuda.cpp convolution_cuda.h convolution_cuda.cu
        padding_cuda.cpp padding_cuda.h padding_cuda.cu)

set_target_properties(
        CudaOperator
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)