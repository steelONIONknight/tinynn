//
// Created by lifan on 2021/1/16.
//

#ifndef DLPROJECT_MAT_H
#define DLPROJECT_MAT_H
#include "allocator.h"
#include <iostream>
#include <memory>

//#if TINYNN_CUDA
#include "cuda_util.h"
//#endif

namespace tinynn
{
//声明CudaMat类，适用于cuda算子
class CudaMat;
//定义一个矩阵类
class Mat
{
public:
    //data表示Mat数据起始地址
    //refcount表示Mat被引用次数
    //allocator先不管，未知其在ncnn中的功能。
    //dims表示数据的维度
    //elempack表示有多少个数据打包在一起, 是一个整数类型
    //elemsize表示elempack * sizeof(data_unit)如float32，4 * elempack，整数类型

    //cstep代表通过一个channel经过的Mat中数据单元的数量
    //对于1d和2d的Mat来说，cstep的值为w和w * h
    //对于3d的Mat来说，cstep的值为每个channel对应的元素的个数+padding的元素数量
    void* data;
    //为了适合于CudaMat之间的通信，refcount所占内存由智能指针管理
    //放弃裸指针
    std::shared_ptr<int> refcount;
    Allocator* allocator;
    int dims;
    int width;
    int height;
    int channel;
    int elempack;
    size_t elemsize;
    size_t cstep;
    Mat();
    Mat(int _width, size_t _elemsize = 4u, Allocator* _allocator = 0);
    Mat(int _width, int _height, size_t _elemsize = 4u, Allocator* _allocator = 0);
    Mat(int _width, int _height, int _channel, size_t _elemsize = 4u, Allocator* _allocator = 0);
    //external packed vector
    Mat(int _width, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator = nullptr);
    //external packed image
    Mat(int _width, int _height, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator = nullptr);
    //external packed dim
    Mat(int _width, int _height, int _channel, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator = nullptr);
    //copy constructor
    Mat(const Mat& m);

    //release
    ~Mat();
    void release();
    size_t total() const;
    bool empty() const;
    //deep copy
    Mat clone(Allocator* _allocator = 0) const;
    //deep copy from other matrix, inplace
    void clone_from(const tinynn::Mat& mat, Allocator* _allocator = 0);
    Mat reshape(int _width, Allocator *_allocator = 0) const;
    Mat reshape(int _width, int _height, Allocator *_allocator = 0) const;
    Mat reshape(int _width, int _height, int _channel, Allocator *_allocator = 0) const;
    void create(int _width, size_t _elemsize = 4u, Allocator *_allocator = 0);
    void create(int _width, int _height, size_t _elemsize = 4u, Allocator *_allocator = 0);
    void create(int _width, int _height, int _channel, size_t _elemsize = 4u, Allocator *_allocator = 0);
    void create(int _width, size_t _elemsize, int _elempack, Allocator *_allocator = 0);
    void create(int _width, int _height, size_t _elemsize, int _elempack, Allocator *_allocator = 0);
    void create(int _width, int _height, int _channel, size_t _elemsize, int _elempack, Allocator *_allocator = 0);

    void create_like(const Mat& m, Allocator* _allocator = nullptr);

    Mat& operator=(const Mat& m);
    //tinynn CudaMat
    //从device侧传送结果到host侧

//#if TINYNN_CUDA
    //device端向host端通信
    void create_like(const CudaMat& cuMat, Allocator* _allocator);
    //从device端向host端拷贝数据
    Mat& operator=(const CudaMat& cuMat);
    Mat(const CudaMat& cuMat);
//#endif

    //data reference
    //channel data reference
    Mat refer_channel(int c);
    const Mat refer_channel(int c) const;

    float* row(int y);
    const float* row(int y) const;

    template<typename T>
    T* row(int y);

    template<typename T>
    const T* row(int y) const;

    //类型转化操作
    template<typename T>
    operator T*();

    template<typename T>
    operator const T*() const;

    float& operator[](size_t n);
    const float& operator[](size_t n) const;



    static Mat from_float16(const unsigned short* data, int size);
};
//#if TINYNN_CUDA
//适用于cuda的Mat数据结构
//目前的开发环境是cuda version 10.2，显卡RTX 2080
//分配的GPU memory按照512Bytes对齐
class CudaMat
{
public:
    void* data;
//    int* refcount;
    std::shared_ptr<int> refcount;
    CudaAllocator* cudaAllocator;
    int dims;
    int width;
    int height;
    int channel;
    int elempack;
    size_t elemsize;
    size_t cstep;
    //分配GPU memory使用的参数
    //每行元素的所占总内存大小（包含填充的空间）
    size_t pitch;

    CudaMat();
    CudaMat(int _width, size_t _elemsize = 4u, CudaAllocator* _cudaAllocator = nullptr);
    CudaMat(int _width, int _height, size_t _elemsize = 4u, CudaAllocator* _cudaAllocator = nullptr);
    CudaMat(int _width, int _height, int _channel, size_t _elemsize = 4u, CudaAllocator* _cudaAllocator = nullptr);
    //external packed vector
    CudaMat(int _width, void* _data, size_t _elemsize, int _elempack, CudaAllocator* _cudaAllocator = nullptr);
    //external packed image
    CudaMat(int _width, int _height, void* _data, size_t _elemsize, int _elempack, CudaAllocator* _cudaAllocator = nullptr);
    //external packed dim
    CudaMat(int _width, int _height, int _channel, void* _data, size_t _elemsize, int _elempack, CudaAllocator* _cudaAllocator = nullptr);
    //copy from host
    //don't know usage of gpu_data_buffer
    //maybe later it's useful
    CudaMat(const Mat& m, CudaAllocator* _cudaAllocator/*, void* gpu_data_buffer*/);
    //copy from device
    CudaMat(const CudaMat& cuMat);

    //release
    ~CudaMat();
    void release();
    size_t total() const;
    bool empty() const;

    CudaMat clone(CudaAllocator* _cudaAllocator = nullptr) const;
    CudaMat reshape(int _width, CudaAllocator* _cudaAllocator = nullptr) const;
    CudaMat reshape(int _width, int _height, CudaAllocator* _cudaAllocator = nullptr) const;
    CudaMat reshape(int _width, int _height, int _channel, CudaAllocator* _cudaAllocator = nullptr) const;
    //TODO
    //create函数需要根据GPU的特点重构
    //使用vector（1d）是，使用cudaMalloc分配GPU memory
    //使用matrix（2d）时，使用cudaMalloc分配GPU memory
    //使用tensor（3d）时，使用cudaMalloc2D分配GPU memory
    //分配GPU memory时，能够保证地址按照512Bytes对齐（当前开发环境下）
    //GPU访存按照128B对齐访问
    void create(int _width, size_t _elemsize = 4u, CudaAllocator* _cudaAllocator = nullptr);
    void create(int _width, int _height, size_t _elemsize = 4u, CudaAllocator* _cudaAllocator = nullptr);
    void create(int _width, int _height, int _channel, size_t _elemsize = 4u, CudaAllocator* _cudaAllocator = nullptr);
    void create(int _width, size_t _elemsize, int _elempack, CudaAllocator* _cudaAllocator = nullptr);
    void create(int _width, int _height, size_t _elemsize, int _elempack, CudaAllocator* _cudaAllocator = nullptr);
    void create(int _width, int _height, int _channel, size_t _elemsize, int _elempack, CudaAllocator* _cudaAllocator = nullptr);

    void create_like(const Mat& m, CudaAllocator* _cudaAllocator = nullptr);

    void create_like(const CudaMat& cuMat, CudaAllocator* _cudaAllocator = nullptr);
//    //data reference
//    CudaMat refer_channel(int c);
//    const CudaMat refer_channel(int c) const;
//
//    //类型转化符
//    template<typename T>
//    operator T*();
//
//    template<typename T>
//    operator const T*() const;
//
//    //override []
//    float& operator[](size_t n);
//
//    const float& operator[](size_t n) const;
//    static CudaMat from_float16(const unsigned short* data, int size);

};
//#endif

unsigned short float32_to_float16(float value);
float float16_to_float32(unsigned short value);

inline Mat::Mat():width(0), height(0), channel(0), elemsize(0), allocator(0), data(0), refcount(0),
                  elempack(0), cstep(0), dims(0) {}

inline Mat::Mat(int _width, size_t _elemsize, Allocator *_allocator):
        width(0), height(0), channel(0), elemsize(0), allocator(0), data(0), refcount(0),
        elempack(0), cstep(0), dims(0)
{
    create(_width, _elemsize, _allocator);
}

inline Mat::Mat(int _width, int _height, size_t _elemsize, Allocator *_allocator):
        width(0), height(0), channel(0), elemsize(0), allocator(0), data(0), refcount(0),
        elempack(0), cstep(0), dims(0)
{
    create(_width, _height, _elemsize, _allocator);
}

inline Mat::Mat(int _width, int _height, int _channel, size_t _elemsize, Allocator *_allocator):
                width(0), height(0), channel(0), elemsize(0), allocator(0), data(0), refcount(0),
                elempack(0), cstep(0), dims(0)
{
    create(_width, _height, _channel, _elemsize, _allocator);
}

inline Mat::Mat(int _width, void *_data, size_t _elemsize, int _elempack, Allocator *_allocator):
                width(_width), height(0), channel(0), elemsize(_elemsize), allocator(nullptr),
                data(_data), refcount(nullptr), elempack(_elempack), dims(1)
{
    cstep = width;
}

inline Mat::Mat(int _width, int _height, void *_data, size_t _elemsize, int _elempack, Allocator *_allocator):
                width(_width), height(_height), channel(1), elemsize(_elemsize), allocator(_allocator),
                data(_data), refcount(nullptr), elempack(_elempack), dims(2)
{
    cstep = (size_t)width * height;
}

inline Mat::Mat(int _width, int _height, int _channel, void *_data, size_t _elemsize, int _elempack, Allocator *_allocator):
                width(_width), height(_height), channel(_channel), elemsize(_elemsize), allocator(_allocator),
                data(_data), refcount(nullptr), elempack(_elempack), dims(3)
{
    cstep = align_size((size_t)width * height * elemsize, 16) / elemsize;
}
//拷贝了对象一次，refcount值必须加1
inline Mat::Mat(const Mat &m):
                width(m.width), height(m.height), channel(m.channel), elemsize(m.elemsize), allocator(m.allocator),
                data(m.data), refcount(m.refcount), elempack(m.elempack), cstep(m.cstep), dims(m.dims)
{
    if (refcount)
        XADD(refcount.get(), 1);
}

inline bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t Mat::total() const
{
    return cstep * channel;
}

inline Mat Mat::clone(Allocator *_allocator) const
{
    if (empty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(width, elemsize, elempack, _allocator);
    else if (dims == 2)
        m.create(width, height, elemsize, elempack, _allocator);
    else
        m.create(width, height, channel, elemsize, elempack, _allocator);

    if (total() > 0)
        memcpy(m.data, data, total() * elemsize);

    return m;
}

inline void Mat::clone_from(const tinynn::Mat &mat, Allocator *_allocator)
{
    *this = mat.clone(_allocator);
}

inline Mat Mat::reshape(int _width, Allocator *_allocator) const
{
    if (width * height * channel != _width)
        return Mat();

    if (dims == 3 && cstep != (size_t)width * height)
    {
        Mat m;
        m.create(_width, elemsize, elempack, _allocator);
        for (int i = 0; i < channel; ++i)
        {
            const void *ptr = (unsigned char*)data + (size_t)i * cstep * elemsize;
            void *mptr = (unsigned char*)m.data + (size_t)i * width * height * elemsize;
            memcpy(mptr, ptr, width * height * elemsize);
        }
        return m;
    }

    Mat m = *this;
    m.dims = 1;
    m.width = _width;
    m.height = 1;
    m.channel = 1;
    m.cstep = _width;

    return m;
}
inline Mat Mat::reshape(int _width, int _height, Allocator *_allocator) const
{
    if (width * height * channel != _width * _height)
        return Mat();

    if (dims == 3 && cstep != (size_t)width * height)
    {
        Mat m;
        m.create(_width, _height, elemsize, elempack, _allocator);
        for (int i = 0; i < channel; ++i)
        {
            const void* ptr = (unsigned char*)data + (size_t)i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * width * height * elemsize;
            memcpy(mptr, ptr, (size_t)width * height * elemsize);
        }
        return m;
    }

    Mat m;
    m = *this;
    m.dims = 2;
    m.width = _width;
    m.height = _height;
    m.channel = 1;
    m.cstep = (size_t)_width * _height;

    return m;
}
inline Mat Mat::reshape(int _width, int _height, int _channel, Allocator *_allocator) const
{
    if (width * height * channel != _width * _height * _channel)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_width * _height != align_size((size_t)_width * _height, 16) / elemsize)
        {
            Mat m;
            m.create(_width, _height, _channel, elemsize, elempack, _allocator);

            for (int i = 0; i < _channel; ++i)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _width * _height * elemsize;
                void* mptr = (unsigned char*)m.data + (size_t)i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_width * _height * elemsize);
            }
            return m;
        }
    }
    else if (channel != _channel)
    {
        Mat temp = reshape(_width * _height * _channel, _allocator);
        return temp.reshape(_width, _height, _channel, _allocator);
    }

    Mat m;
    m = *this;
    m.dims = 3;
    m.width = _width;
    m.height = _height;
    m.channel = _channel;
    m.cstep = align_size((size_t)_width * _height * elemsize, 16) / elemsize;

    return m;
}
//创建vector 1d
inline void Mat::create(int _width, size_t _elemsize, Allocator *_allocator)
{
    if (dims == 1 && width == _width && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;
    release();

    allocator = _allocator;
    dims = 1;
    width = _width;
    height = 1;
    channel = 1;
    elemsize = _elemsize;
    elempack = 1;

    cstep = _width;
    if (total() > 0)
    {
        //传入参数_elemsize可能不是4的倍数
        size_t total_sz = align_size(total() * elemsize, 4);
        if (allocator)
            std::cout << "TODO" << std::endl;
            //TODO
        else
            data = align_malloc(total_sz);
        refcount = std::make_shared<int>(1);
    }
}

//创建matrix 2d
inline void Mat::create(int _width, int _height, size_t _elemsize, Allocator *_allocator)
{
    if (dims == 2 && width == _width && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();
    allocator = _allocator;
    dims = 2;
    width = _width;
    height = _height;
    channel = 1;
    elemsize = _elemsize;
    elempack = 1;

    cstep = (size_t)width * height;

    if (total() > 0)
    {
        size_t total_sz = align_size(total() * elemsize, 4);
        if (allocator)
            std::cout<<"to do"<<std::endl;
            //TODO
        else
            data = align_malloc(total_sz);
        refcount = std::make_shared<int>(1);
    }

}

//创建tensor 3d
//elempack作用
//w = 2, h = 3, c = 4, elempack = 1, float32 elemsize = 4 * elempack
//每个channel按照16字节对齐，多余的空间为padding，例子的情况为2个padding
//内存布局
//############################################
//# 0  # 1  # 2  # 3  # 4  # 5  # pad # pad ##
//############################################
//# 6  # 7  # 8  # 9  # 10 # 11 # pad # pad ##
//############################################
//# 12 # 13 # 14 # 15 # 16 # 17 # pad # pad ##
//############################################
//# 18 # 19 # 20 # 21 # 22 # 23 # pad # pad ##
//############################################

//w = 2, h = 3, c = 1, elempack = 4, float32 elemsize = 4 * elempack
//####################################################################
//# pack0  # pack1  # pack2  # pack3  # pack4  # pack5  # pad # pad ##
//####################################################################
//pack0对应(0, 6, 12, 18)
//这种处理针对x86的AVX(elempack = 8)，SSE2(elempack = 4)等指令优化，使CPU同时处理多个数据
inline void Mat::create(int _width, int _height, int _channel, size_t _elemsize, Allocator *_allocator)
{
    if (dims == 3 && width == _width && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();
    allocator = _allocator;
    dims = 3;
    width = _width;
    height = _height;
    channel = _channel;
    elemsize = _elemsize;
    elempack = 1;

    //注意每个channel的大小按照16字节对齐
    cstep = align_size((size_t)width * height * elemsize, 16) / elemsize;
    if (total() > 0)
    {
        size_t total_sz = align_size(total() * elemsize , 4);
        if (allocator)
            std::cout<<"to do"<<std::endl;
            //TODO
        else
            data = align_malloc(total_sz);
        refcount = std::make_shared<int>(1);
    }
}
inline void Mat::create(int _width, size_t _elemsize, int _elempack, Allocator *_allocator)
{
    if (dims == 1 && width == _width && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;
    release();

    width = _width;
    height = 1;
    channel = 1;
    dims = 1;
    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    cstep = _width;
    if (total() > 0)
    {
        size_t total_sz = align_size(total() * elemsize, 4);
        if (allocator)
            std::cout << "TODO" << std::endl;
            //TODO
        else
            data = align_malloc(total_sz);
        refcount = std::make_shared<int>(1);
    }
}

inline void Mat::create(int _width, int _height, size_t _elemsize, int _elempack, Allocator *_allocator)
{
    if (dims == 2 && width == _width && height == _height && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;
    release();

    width = _width;
    height = _height;
    channel = 1;
    dims = 2;

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;
    cstep = (size_t)width * height;

    if (total() > 0)
    {
        size_t total_sz = align_size(total() * elemsize, 4);
        if (allocator)
            std::cout << "TODO" << std::endl;
            //TODO
        else
            data = align_malloc(total_sz);
        refcount = std::make_shared<int>(1);
    }
}

inline void Mat::create(int _width, int _height, int _channel, size_t _elemsize, int _elempack, Allocator *_allocator)
{
    if (dims == 3 && width == _width && height == _height && channel == _channel && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;
    release();

    width = _width;
    height = _height;
    channel = _channel;
    dims = 3;

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;
    //注意每个channel的大小按照16字节对齐
    cstep = align_size((size_t)width * height * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t total_sz = align_size(total() * elemsize, 4);
        if (allocator)
            std::cout << "TODO" << std::endl;
            //TODO
        else
            data = align_malloc(total_sz);
        refcount = std::make_shared<int>(1);
    }
}

template<typename T>
inline Mat::operator T*() {
    return (T*)data;
}

template<typename T>
inline Mat::operator const T*() const{
    return (const T*)data;
}

inline float &Mat::operator[](size_t n) {
    return *((float *)data + n);
}
inline const float &Mat::operator[](size_t n) const {
    return *((const float *)data + n);
}
inline Mat::~Mat() {
    release();
}
//释放所有资源
inline void Mat::release() {
    if (refcount && XADD(refcount.get(), -1) == 1)
    {
        if (allocator)
            std::cout << "TODO" << std::endl;
        else
            align_free(data);
    }
    //data置零，防止野指针
    data = 0;
    dims = 0;
    width = 0;
    height = 0;
    channel = 0;

    elemsize = 0;
    elempack = 0;
    cstep = 0;
    refcount = 0;
}

inline Mat Mat::refer_channel(int c)
{
    return Mat(width, height, (unsigned char*)data + c * cstep * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::refer_channel(int c) const
{
    return Mat(width, height, (unsigned char*)data + c * cstep * elemsize, elemsize, elempack, allocator);
}

inline float *Mat::row(int y)
{
    return (float*)((unsigned char*)data + (size_t)y * width * elemsize);
}

inline const float *Mat::row(int y) const
{
    return (const float*)((unsigned char*)data + (size_t)y * width * elemsize);
}

template<typename T>
inline T* Mat::row(int y)
{
    return (T*)((unsigned char*)data + (size_t)y * width * elemsize);
}

template<typename T>
inline const T* Mat::row(int y) const
{
    return (const T*)((unsigned char*)data + (size_t)y * width * elemsize);
}

inline void Mat::create_like(const Mat &m, Allocator *_allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.width, m.elemsize, m.elempack, _allocator);
    else if (_dims == 2)
        create(m.width, m.height, m.elemsize, m.elempack, _allocator);
    else
        create(m.width, m.height, m.channel, m.elemsize, m.elempack, _allocator);

}

inline Mat &Mat::operator=(const Mat &m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        XADD(m.refcount.get(), 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    width = m.width;
    height = m.height;
    channel = m.channel;

    cstep = m.cstep;

    return *this;
}

//#if TINYNN_CUDA
inline void Mat::create_like(const CudaMat &cuMat, Allocator *_allocator)
{
    int _dims = cuMat.dims;
    if (_dims == 1)
        create(cuMat.width, cuMat.elemsize, cuMat.elempack, _allocator);
    else if (_dims == 2)
        create(cuMat.width, cuMat.height, cuMat.elemsize, cuMat.elempack, _allocator);
    else
        create(cuMat.width, cuMat.height, cuMat.channel, cuMat.elemsize, cuMat.elempack, _allocator);

}
//TODO
//maybe something wrong
inline Mat &Mat::operator=(const CudaMat &cuMat)
{
    release();

    create_like(cuMat, nullptr);

    if (dims == 1)
    {
        CHECK(cudaMemcpy(data, cuMat.data, cuMat.width * cuMat.elemsize, cudaMemcpyDeviceToHost));
    }
    else if (dims == 2)
    {
        CHECK(cudaMemcpy(data, cuMat.data, cuMat.width * cuMat.height * cuMat.elemsize, cudaMemcpyDeviceToHost));
    }
    else
    {
        CHECK(cudaMemcpy2D(data, cstep * elemsize, cuMat.data, cuMat.pitch, (size_t)cuMat.width * cuMat.height * cuMat.elemsize, cuMat.channel, cudaMemcpyDeviceToHost));
    }

    return *this;
}

//TODO
//maybe something wrong
inline Mat::Mat(const CudaMat &cuMat)
{
    create_like(cuMat, nullptr);
    if (dims == 1)
    {
        CHECK(cudaMemcpy(data, cuMat.data, cuMat.width * cuMat.elemsize, cudaMemcpyDeviceToHost));
    }
    else if (dims == 2)
    {
        CHECK(cudaMemcpy(data, cuMat.data, cuMat.width * cuMat.height * cuMat.elemsize, cudaMemcpyDeviceToHost));
    }
    else
    {
        CHECK(cudaMemcpy2D(data, cstep * elemsize, cuMat.data, cuMat.pitch, (size_t)cuMat.width * cuMat.height * cuMat.elemsize, cuMat.channel, cudaMemcpyDeviceToHost));
    }
}

//#endif

//CudaMat
//#if TINYNN_CUDA

inline CudaMat::CudaMat(): width(0), height(0), channel(0), dims(0), elemsize(0), elempack(0),
        refcount(0), cudaAllocator(nullptr), cstep(0), data(nullptr), pitch(0)
{
}

inline CudaMat::CudaMat(int _width, size_t _elemsize, CudaAllocator *_cudaAllocator)
{
    create(_width, _elemsize, _cudaAllocator);
}

inline CudaMat::CudaMat(int _width, int _height, size_t _elemsize, CudaAllocator *_cudaAllocator)
{
    create(_width, _height, _elemsize, _cudaAllocator);
}

inline CudaMat::CudaMat(int _width, int _height, int _channel, size_t _elemsize, CudaAllocator *_cudaAllocator)
{
    create(_width, _height, _channel, _elemsize, _cudaAllocator);
}

inline CudaMat::CudaMat(int _width, void *_data, size_t _elemsize, int _elempack, CudaAllocator *_cudaAllocator):
    data(_data), width(_width), height(1), channel(1), dims(1), elemsize(_elemsize), elempack(_elempack),
    cudaAllocator(_cudaAllocator), refcount(0), pitch(0)
{
    cstep = width;
}

inline CudaMat::CudaMat(int _width, int _height, void *_data, size_t _elemsize, int _elempack, CudaAllocator *_cudaAllocator):
        data(_data), width(_width), height(_height), channel(1), dims(2), elemsize(_elemsize), elempack(_elempack),
        cudaAllocator(_cudaAllocator), refcount(0), pitch(0)
{
    cstep = width * height;
}

inline CudaMat::CudaMat(int _width, int _height, int _channel, void *_data, size_t _elemsize, int _elempack, CudaAllocator *_cudaAllocator):
        data(_data), width(_width), height(_height), channel(_channel), dims(3), elemsize(_elemsize), elempack(_elempack),
        cudaAllocator(_cudaAllocator), refcount(0), pitch(0)
{
    cstep = align_size((size_t)width * height * elemsize, 16) / elemsize;
}

inline CudaMat::CudaMat(const Mat &m, CudaAllocator *_cudaAllocator):width(m.width), height(m.height), channel(m.channel),
        dims(m.dims), elemsize(m.elemsize), elempack(m.elempack), cudaAllocator(_cudaAllocator), refcount(0), cstep(m.cstep)
{
    if (total() > 0)
    {
        if (dims == 1)
        {
            data = cudaAllocator->align_malloc(width, elemsize);
            CHECK(cudaMemcpy(data, m.data, (size_t)width * elemsize, cudaMemcpyHostToDevice));
        }
        else if (dims == 2)
        {
            data = cudaAllocator->align_malloc(width, height, elemsize);
            CHECK(cudaMemcpy(data, m.data, (size_t)width * height * elemsize, cudaMemcpyHostToDevice));
        }
        else
        {
            data = cudaAllocator->align_malloc(width, height, channel, elemsize, &pitch, cstep);
            CHECK(cudaMemcpy2D(data, pitch, m.data, m.cstep * m.elemsize, (size_t)m.width * m.height * m.elemsize, m.channel, cudaMemcpyHostToDevice));
        }
        refcount = std::make_shared<int>(1);
    }
}

inline CudaMat::~CudaMat()
{
    release();
}

inline void CudaMat::release()
{
    if (refcount && XADD(refcount.get(), -1) == 1)
        cudaAllocator->align_free(data);

    data = nullptr;

    width = 0;
    height = 0;
    channel = 0;
    dims = 0;
    elemsize = 0;
    elempack = 0;
    cstep = 0;
    cudaAllocator = nullptr;
    refcount = 0;
    pitch = 0;
}

inline size_t CudaMat::total() const
{
    return cstep * channel;
}

inline bool CudaMat::empty() const
{
    return data == nullptr || total() == 0;
}
//TODO
//some problems
inline CudaMat CudaMat::clone(CudaAllocator *_cudaAllocator) const
{
    if (!empty())
        return CudaMat();

    CudaMat cuMat;
    if (dims == 1)
    {
        cuMat.create(width, elemsize, elempack, _cudaAllocator);
        CHECK(cudaMemcpy(cuMat.data, data,  (size_t)width * elemsize, cudaMemcpyDeviceToDevice));
    }
    else if (dims == 2)
    {
        cuMat.create(width, height, elemsize, elempack, _cudaAllocator);
        CHECK(cudaMemcpy(cuMat.data, data, (size_t)width * height * elemsize, cudaMemcpyDeviceToDevice));
    }
    else
    {
        cuMat.create(width, height, channel, elemsize, elempack, _cudaAllocator);
        CHECK(cudaMemcpy2D(cuMat.data, pitch, data, pitch, (size_t)width * height * elemsize, channel, cudaMemcpyDeviceToDevice));
    }
    return cuMat;
}

inline CudaMat CudaMat::reshape(int _width, CudaAllocator *_cudaAllocator) const
{
//TODO
}

inline CudaMat CudaMat::reshape(int _width, int _height, CudaAllocator *_cudaAllocator) const
{
//TODO
}

inline CudaMat CudaMat::reshape(int _width, int _height, int _channel, CudaAllocator *_cudaAllocator) const
{
//TODO
}

inline void CudaMat::create(int _width, size_t _elemsize, CudaAllocator *_cudaAllocator)
{
    if (dims == 1 && width == _width && elemsize == _elemsize && elempack == 1 && cudaAllocator == _cudaAllocator)
        return;

    release();

    cudaAllocator = _cudaAllocator;
    width = _width;
    height = 1;
    channel = 1;
    dims = 1;
    elemsize = _elemsize;
    elempack = 1;

    cstep = width;

    if (total() > 0)
    {
//        size_t total_sz = align_size(total() * elemsize, 4);
//        data = cudaAllocator->align_malloc(total_sz + (int)sizeof(*refcount));
        data = cudaAllocator->align_malloc(width, elemsize);
        refcount = std::make_shared<int>(1);
    }
}

inline void CudaMat::create(int _width, int _height, size_t _elemsize, CudaAllocator *_cudaAllocator)
{
    if (dims == 2 && width == _width && height == _height && elemsize == _elemsize && elempack == 1 && cudaAllocator == _cudaAllocator)
        return;

    release();

    cudaAllocator = _cudaAllocator;
    width = _width;
    height = _height;
    channel = 1;
    dims = 2;
    elemsize = _elemsize;
    elempack = 1;

    cstep = width * height;

    if (total() > 0)
    {
//        size_t total_sz = align_size((size_t)total() * elemsize, 4);
//        data = cudaAllocator->align_malloc(total_sz + (int) sizeof(*refcount));
        data = cudaAllocator->align_malloc(width, height, elemsize);
        refcount = std::make_shared<int>(1);
    }
}

inline void CudaMat::create(int _width, int _height, int _channel, size_t _elemsize, CudaAllocator *_cudaAllocator)
{
    if (dims == 3 && width == _width && height == _height && channel == _channel && elemsize == _elemsize && elempack == 1 && cudaAllocator == _cudaAllocator)
        return;

    release();

    cudaAllocator = _cudaAllocator;
    width = _width;
    height = _height;
    channel = _channel;
    dims = 3;
    elemsize = _elemsize;
    elempack = 1;

    cstep = align_size((size_t)width * height * elemsize, 16) / elemsize;

    if (total() > 0)
    {
//        size_t total_sz = align_size((size_t)total() * elemsize, 4);
//        data = cudaAllocator->align_malloc(total_sz + (int) sizeof(*refcount));
        data = cudaAllocator->align_malloc(width, height, channel, elemsize, &pitch, cstep);
        refcount = std::make_shared<int>(1);
    }
}

inline void CudaMat::create(int _width, size_t _elemsize, int _elempack, CudaAllocator *_cudaAllocator)
{
    if (dims == 1 && width == _width && elemsize == _elemsize && elempack == _elempack && cudaAllocator == _cudaAllocator)
        return;

    release();

    cudaAllocator = _cudaAllocator;
    width = _width;
    height = 1;
    channel = 1;
    dims = 1;
    elemsize = _elemsize;
    elempack = _elempack;

    cstep = width;

    if (total() > 0)
    {
//        size_t total_sz = align_size(total() * elemsize, 4);
//        data = cudaAllocator->align_malloc(total_sz + (int)sizeof(*refcount));
        data = cudaAllocator->align_malloc(width, elemsize);
        refcount = std::make_shared<int>(1);
    }
}

inline void CudaMat::create(int _width, int _height, size_t _elemsize, int _elempack, CudaAllocator *_cudaAllocator)
{
    if (dims == 2 && width == _width && height == _height && elemsize == _elemsize && elempack == _elempack && cudaAllocator == _cudaAllocator)
        return;

    release();

    cudaAllocator = _cudaAllocator;
    width = _width;
    height = _height;
    channel = 1;
    dims = 2;
    elemsize = _elemsize;
    elempack = _elempack;

    cstep = width * height;

    if (total() > 0)
    {
//        size_t total_sz = align_size((size_t)total() * elemsize, 4);
//        data = cudaAllocator->align_malloc(total_sz + (int) sizeof(*refcount));
        data = cudaAllocator->align_malloc(width, height, elemsize);
        refcount = std::make_shared<int>(1);
    }
}

inline void CudaMat::create(int _width, int _height, int _channel, size_t _elemsize, int _elempack, CudaAllocator *_cudaAllocator)
{
    if (dims == 3 && width == _width && height == _height && channel == _channel && elemsize == _elemsize && elempack == _elempack && cudaAllocator == _cudaAllocator)
        return;

    release();

    cudaAllocator = _cudaAllocator;
    width = _width;
    height = _height;
    channel = _channel;
    dims = 3;
    elemsize = _elemsize;
    elempack = _elempack;

    cstep = align_size(width * height * elemsize, 16) / elemsize;

    if (total() > 0)
    {
//        size_t total_sz = align_size((size_t)total() * elemsize, 4);
//        data = cudaAllocator->align_malloc(total_sz + (int) sizeof(*refcount));
        data = cudaAllocator->align_malloc(width, height, channel, elemsize, &pitch, cstep);
        refcount = std::make_shared<int>(1);
    }
}

inline void CudaMat::create_like(const Mat &m, CudaAllocator *_cudaAllocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.width, m.elemsize, m.elempack, _cudaAllocator);
    else if (_dims == 2)
        create(m.width, m.height, m.elemsize, m.elempack, _cudaAllocator);
    else
        create(m.width, m.height, m.channel, m.elemsize, m.elempack, _cudaAllocator);

}

inline void CudaMat::create_like(const CudaMat &cuMat, CudaAllocator *_cudaAllocator)
{
    int _dims = cuMat.dims;
    if (_dims == 1)
        create(cuMat.width, cuMat.elemsize, cuMat.elempack, _cudaAllocator);
    else if (_dims == 2)
        create(cuMat.width, cuMat.height, cuMat.elemsize, cuMat.elempack, _cudaAllocator);
    else
        create(cuMat.width, cuMat.height, cuMat.channel, cuMat.elemsize, cuMat.elempack, _cudaAllocator);
}
//#endif
}
#endif //DLPROJECT_MAT_H
