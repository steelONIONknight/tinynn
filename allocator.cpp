//
// Created by lifan on 2021/3/6.
//

#include "allocator.h"

#include "cuda_util.h"

namespace tinynn
{

Allocator::~Allocator()
{
}

CudaAllocator::~CudaAllocator()
{
}

CudaAllocator::CudaAllocator(const CudaDevice *_cudev): cudev(_cudev)
{
    CHECK(cudaSetDevice(cudev->device_index));
}

void *CudaAllocator::align_malloc(size_t size)
{
    void* buffer = nullptr;
    CHECK(cudaMalloc((void**)&buffer, size));

    return buffer;
}

void CudaAllocator::align_free(void *ptr)
{
    if (ptr)
    {
        CHECK(cudaFree(ptr));
    }
}

void *CudaAllocator::align_malloc(int width, size_t elemsize)
{
    void* buffer = nullptr;
    CHECK(cudaMalloc((void**)&buffer, (size_t)width * elemsize));

    return buffer;
}

void *CudaAllocator::align_malloc(int width, int height, size_t elemsize)
{
    void* buffer = nullptr;

    CHECK(cudaMalloc((void**)&buffer, (size_t)width * height * elemsize));

    return buffer;
}

void *CudaAllocator::align_malloc(int width, int height, int channel, size_t elemsize, size_t* pitch, size_t cstep)
{
    void* buffer = nullptr;
    CHECK(cudaMallocPitch((void**)&buffer, pitch, cstep * elemsize, channel));

    return buffer;
}

}//namespace tinynn