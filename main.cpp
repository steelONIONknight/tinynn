//
// Created by lifan on 2021/1/16.
//
#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <vector>
#include "mat.h"
#include "gpu.h"
#include "./layer/cuda/test_cuda.h"

#define MALLOC_ALIGN 16

void print_matrix(tinynn::Mat& matrix)
{
    int len = matrix.width * matrix.height;
    for (int i = 0; i < matrix.channel; ++i)
    {
        float *p = (float*)matrix.refer_channel(i);
        for (int j = 0; j < len; ++j)
        {
            std::cout << p[j] << " ";
        }
        std::cout << std::endl;
    }
}
int main() {
//    unsigned char* test = (unsigned char*)tinynn::align_malloc(100);
//    size_t addr = (size_t)test;
//    std::cout << "addr is " << addr << std::endl;
//    if (addr % MALLOC_ALIGN == 0)
//        std::cout << "Right" << std::endl;
//    else
//        std::cout << "Wrong" << std::endl;
//    tinynn::align_free(test);

    tinynn::Mat matrix(2, 3, 15);
    int len = matrix.width * matrix.height;
    for (int i = 0; i < matrix.channel; ++i)
    {
        float *p = (float *)matrix.refer_channel(i);
        for (int j = 0; j < len; ++j)
        {
            p[j] = (float)(i * len + (j + 1));
        }
    }

    print_matrix(matrix);

    int device_index = 0;
    tinynn::CudaDevice device(device_index);

    tinynn::CudaAllocator cudaAllocator(&device);

    tinynn::CudaMat cuMat(matrix, &cudaAllocator);

    tinynn::fun((float*)cuMat.data, cuMat.width, cuMat.height, cuMat.channel, cuMat.pitch);

    tinynn::Mat matrix2;

    matrix2 = cuMat;

    print_matrix(matrix2);


    return 0;
}
