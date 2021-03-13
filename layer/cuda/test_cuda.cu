//
// Created by lifan on 2021/3/13.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include "test_cuda.h"

__global__ void test3D(float* input, int width, int height, int channel, int pitch = 0)
{
    int idx;
    printf("------------------------------------------------\n");
    printf("print from GPU\n");
    for (int k = 0; k < channel; k++)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                idx = i * width + j;
                printf("%f ", *(float*)((char*)(input + idx) + k * pitch));
            }
            printf("\n");
        }
        printf("----------------------------------------------------\n");
    }
}

namespace tinynn
{

void fun(float* input, int width, int height, int channel, int pitch)
{
    test3D<<<1, 1>>>(input, width, height, channel, pitch);
    cudaDeviceSynchronize();
}
}//namespace tinynn