//
// Created by lifan on 2021/1/16.
//

#ifndef DLPROJECT_ALLOCATOR_H
#define DLPROJECT_ALLOCATOR_H
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include "gpu.h"
#define MALLOC_ALIGN 16

// toy inference engine
namespace tinynn
{
//
static inline size_t align_size(size_t sz, size_t n)
{
    return (sz + n - 1) & -n;
}
template<typename _Tp>
static inline _Tp* align_ptr(_Tp* ptr, size_t size)
{
    //计算返回大于等于ptr存放的地址且是size最小倍数的值
    //(size_t)return_val % align_size == 0
    return (_Tp*)align_size((size_t)ptr, size);

}
static inline void* align_malloc(size_t size)
{
#if (defined(__unix__)) && _POSIX_C_SOURCE >= 200112L
    void* ptr;

    if (posix_memalign(&ptr, MALLOC_ALIGN, size))
        ptr = 0;

    return ptr;
#else
    //申请内存空间的大小 size + 8 + 16 == size + 24 Bytes
    //8Bytes用来存放起始的内存地址，16Bytes作为偏移量，用于内存对齐
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (udata == 0)
        return 0;
    //注意(unsigned char**)udata + 1是udata保存的地址 + 8 == sizeof(unsigned char*)
    unsigned char** adata = align_ptr((unsigned char**)udata + 1, MALLOC_ALIGN);
    //保存原始分配内存的地址
    adata[-1] = udata;
    return adata;
#endif
}
static inline void align_free(void* ptr)
{
    if (ptr)
    {
#if (defined(__unix__)) &&_POSIX_C_SOURCE >= 200112L
      free(ptr);
#else
        unsigned char** addr = (unsigned char**)ptr - 1;
        free(*addr);
        return;
#endif
    }
}

//TODO
class Allocator
{
public:
    virtual ~Allocator();
    virtual void* align_malloc(size_t size) = 0;
    virtual void align_free(void* ptr) = 0;
};

//class CudaDevice;
//TODO
class CudaAllocator: public Allocator
{
public:
    CudaAllocator(const CudaDevice* _cudev);
    virtual ~CudaAllocator();
    //此函数不要使用
    virtual void* align_malloc(size_t size);

    virtual void* align_malloc(int width, size_t elemsize);
    virtual void* align_malloc(int width, int height, size_t elemsize);
    virtual void* align_malloc(int width, int height, int channel, size_t elemsize, size_t* pitch, size_t cstep);
    virtual void align_free(void* ptr);

public:
    const CudaDevice* cudev;
};

#define XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
}
#endif //DLPROJECT_ALLOCATOR_H
