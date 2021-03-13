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
    cudaGetDeviceProperties(&cudaProp, device_index);
    printf("> now using device %d: %s \n", device_index, cudaProp.name);
    printf("> number of multi-processor is %d\n", cudaProp.multiProcessorCount);
}

CudaDevice::~CudaDevice()
{
}

}//namespace tinynn