//
// Created by lifan on 2021/3/5.
//
//#if TINYNN_CUDA
#ifndef DLPROJECT_CUDA_UTIL_H
#define DLPROJECT_CUDA_UTIL_H
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}


#endif //DLPROJECT_CUDA_UTIL_H
//#endif