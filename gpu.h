//
// Created by lifan on 2021/3/6.
//

#ifndef TINYNN_GPU_H
#define TINYNN_GPU_H
//#if TINYNN_CUDA
#include "cuda_util.h"

namespace tinynn
{
class CudaDevice
{
public:
    CudaDevice(int _device_index);
    ~CudaDevice();

public:
    int device_index;
};
int get_cuda_device_count();
void get_cuda_device_info();

}//namespace tinynn
//#endif
#endif //TINYNN_GPU_H
