//
// Created by lifan on 2021/3/6.
//

#include "gpu.h"
#include <iostream>
namespace tinynn
{

CudaDevice::CudaDevice(int _device_index)
{
    device_index = _device_index;
    CHECK(cudaSetDevice(device_index));
//    cudaGetDeviceProperties(&cudaProp, device_index);
//    printf("> now using device %d: %s \n", device_index, cudaProp.name);
//    printf("> number of multi-processor is %d\n", cudaProp.multiProcessorCount);
}

CudaDevice::~CudaDevice()
{
}

int get_cuda_device_count()
{
    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count));
    return device_count;
}
void get_cuda_device_info()
{
    int device_count = get_cuda_device_count();
    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp cudaProp;
        CHECK(cudaGetDeviceProperties(&cudaProp, i));
        float memory_bandwidth = (cudaProp.memoryClockRate * 1000.f * (cudaProp.memoryBusWidth / 8.f) * 2) / 1000000000.f;

        printf("> Cuda Device number: %d\n", device_count);
        printf("> Device name: %s\n", cudaProp.name);
        printf("> Device Memory Clock Rate (KHz): %d\n", cudaProp.memoryClockRate);
        printf("> Device Memory Bus Width (bits): %d\n", cudaProp.memoryBusWidth);
        printf("> Device Peak Memory Bandwidth (GB/s): %.2f\n", memory_bandwidth);
        printf("> Shared memory available per block(Bytes): %zu\n", cudaProp.sharedMemPerBlock);
        printf("> Multi-processor number: %d\n", cudaProp.multiProcessorCount);
    }
}
}//namespace tinynn