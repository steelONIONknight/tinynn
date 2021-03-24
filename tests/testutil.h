//
// Created by lifan on 2021/3/14.
//

#ifndef TINYNN_TESTUTIL_H
#define TINYNN_TESTUTIL_H
#include "../cpu.h"
#include "../layer.h"
#include "../mat.h"
#include "prng.h"
#include "../gpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND() prng_rand(&g_prng_rand_state)

static float RandomFloat(float a = -1.2f, float b = 1.2f)
{
    float random = ((float)RAND()) / (float)uint64_t(-1);
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

static void Randomize(tinynn::Mat& m, float a = -1.2f, float b = 1.2f)
{
    for (size_t i = 0; i < m.total(); ++i)
    {
        m[i] = RandomFloat(a, b);
    }
}

static tinynn::Mat RandomMat(int width)
{
    tinynn::Mat m(width);
    Randomize(m);
    return m;
}

static tinynn::Mat RandomMat(int width, int height)
{
    tinynn::Mat m(width, height);
    Randomize(m);
    return m;
}

static tinynn::Mat RandomMat(int width, int height, int channel)
{
    tinynn::Mat m(width, height, channel);
    Randomize(m);
    return m;
}



static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);

    if (diff <= epsilon)
        return true;

    //relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

static int Compare(const tinynn::Mat& a, const tinynn::Mat& b, float epsilon = 0.001)
{
#define CHECK_MEMBER(m)                                                                \
    if (a.m != b.m)                                                                    \
    {                                                                                  \
        fprintf(stderr, #m " not match    expect %d but got %d\n", (int)a.m, (int)b.m);\
        return -1;                                                                     \
    }

    CHECK_MEMBER(dims)
    CHECK_MEMBER(width)
    CHECK_MEMBER(height)
    CHECK_MEMBER(channel)
    CHECK_MEMBER(elemsize)
    CHECK_MEMBER(elempack)
#undef CHECK_MEMBER

    for (int q = 0; q < a.channel; ++q)
    {
        const tinynn::Mat ma = a.refer_channel(q);
        const tinynn::Mat mb = b.refer_channel(q);
        for (int i = 0; i < a.height; ++i)
        {
            const float* pa = a.row(i);
            const float* pb = b.row(i);
            for (int j = 0; j < a.width; ++j)
            {
                if (!NearlyEqual(pa[j], pb[j], epsilon))
                {
                    fprintf(stderr, "value not match  at channel:%d height:%d width:%d    expect %f but got %f\n", q, i, j, pa[j], pb[j]);
                    return -1;
                }
            }
        }
    }
    return 0;
}
static int CompareMat(const tinynn::Mat& a, const tinynn::Mat& b, float epsilon = 0.001)
{
    if (a.elemsize == 2u)
    {
        //TODO
        tinynn::Mat a32;

    }

    if (b.elemsize == 2u)
    {
        //TODO
        tinynn::Mat b32;

    }

    return Compare(a, b, epsilon);
}

static int CompareMat(const std::vector<tinynn::Mat>& a, const std::vector<tinynn::Mat>& b, float epsilon = 0.001)
{
    if (a.size() != b.size())
    {
        fprintf(stderr, "output blob count not match %zu %zu\n", a.size(), b.size());
        return -1;
    }

    for (size_t i = 0; i < a.size(); ++i)
    {
        if (CompareMat(a[i], b[i], epsilon))
        {
            fprintf(stderr, "output blob %zu not match\n", i);
            return -1;
        }
    }

    return 0;
}

template<typename T>
int test_layer_cuda(int type_index, const tinynn::ParamDict&pd, const std::vector<tinynn::Mat>& weights, const tinynn::Option& _opt, const tinynn::Mat& a, tinynn::Mat& d, const tinynn::Mat& top_shape, void(*func)(T*))
{
    tinynn::Layer* op = tinynn::create_layer(type_index);

    if (!op->support_cuda)
    {
        delete op;
        return 233;
    }
    //set GPU
    int device_index = 0;
    tinynn::CudaDevice cudev(device_index);
    tinynn::CudaAllocator cudaAllocator(&cudev);

    if (func)
    {
        (*func)((T*)op);
    }

    if (top_shape.dims)
    {
        op->bottom_shapes.resize(1);
        op->top_shapes.resize(1);
        op->bottom_shapes[0] = a;
        op->top_shapes[0] = top_shape;
    }
    op->load_parm(pd);

    tinynn::CudaModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    tinynn::Option opt = _opt;
    opt.num_threads = 1;
    opt.use_cuda_compute = true;

    opt.workspace_cuda_allocator = &cudaAllocator;

    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_fp16_packed = false;

    op->create_pipeline(opt);

    tinynn::Mat a2 = a;

    tinynn::CudaMat a_gpu(a2, &cudaAllocator);
    tinynn::CudaMat d_gpu;
    d_gpu.create_like(a_gpu, &cudaAllocator);


    std::chrono::high_resolution_clock::time_point begin, end;

    if (op->support_inplace)
    {
        begin = std::chrono::high_resolution_clock::now();
        op->forward_inplace(a_gpu, opt);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        CHECK(cudaGetLastError());
        d = a_gpu;
    }
    else
    {
        begin = std::chrono::high_resolution_clock::now();
        op->forward(a_gpu, d_gpu, opt);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        CHECK(cudaGetLastError());
        d = d_gpu;
    }
    std::chrono::duration<double, std::milli> fp_ms = end - begin;
    printf("test_layer_cuda execution time: %lf ms\n", fp_ms.count());

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

template<typename T>
int test_layer_naive(int type_index, const tinynn::ParamDict& pd, const std::vector<tinynn::Mat>& weights, const tinynn::Mat& input, tinynn::Mat& output, void(*func)(T*))
{
    tinynn::Layer* op = tinynn::create_layer(type_index);

    if (func)
    {
        (*func)((T*)op);
    }

    op->load_parm(pd);

    tinynn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    tinynn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_shader_pack8 = false;

    op->create_pipeline(opt);

    if (op->support_inplace)
    {
        output = input.clone();
        ((T*)op)->T::forward_inplace(output, opt);
    }
    else
    {
        ((T*)op)->T::forward(input, output, opt);
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

template<typename T>
int test_layer(int type_index, const tinynn::ParamDict& pd, const std::vector<tinynn::Mat>& weights, const tinynn::Option& opt, const tinynn::Mat& a, const tinynn::Mat& top_shape = tinynn::Mat(), float epsilon = 0.001f, void(*func)(T*) = 0)
{
    tinynn::Mat b;
    {
        int ret = test_layer_naive(type_index, pd, weights, a, b, func);
        if (ret != 0)
        {
            fprintf(stderr, "test_layer_naive failed\n");
            return -1;
        }
    }

    //cuda
    {
        tinynn::Mat d;
        int ret = test_layer_cuda(type_index, pd, weights, opt, a, d, tinynn::Mat(), func);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cuda failed\n");
            return -1;
        }
    }

    return 0;
}

template<typename T>
int test_layer(const char* layer_type, const tinynn::ParamDict& pd, const std::vector<tinynn::Mat>& weights, const tinynn::Mat& input, float epsilon = 0.001f, void (*func)(T*) = 0)
{
    tinynn::Option opts[2];

    opts[0].use_packing_layout = false;
    opts[0].use_fp16_arithmetic = false;
    opts[0].use_fp16_storage = false;
    opts[0].use_fp16_packed = false;
    opts[0].use_shader_pack8 = false;


    //TODO
    //opts[1] set


    for (int i = 0; i < 1; ++i)
    {
        const tinynn::Option& opt = opts[i];

        tinynn::Mat input_fp16;
        std::vector<tinynn::Mat> weights_fp16;
        float epsilon_fp16;

        if (opt.use_fp16_packed || opt.use_fp16_storage)
        {
            //TODO
        }
        else
        {
            input_fp16 = input;
            weights_fp16 = weights;
            epsilon_fp16 = epsilon;
        }

        if (opt.use_fp16_arithmetic)
        {
            epsilon_fp16 = epsilon * 1000;//1.0
        }

        tinynn::Mat top_shape;
        int ret = test_layer<T>(tinynn::layer_to_index(layer_type), pd, weights, opt, input, top_shape, epsilon, func);
        if (ret != 0)
        {
            fprintf(stderr, "test_layer %s failed\n", layer_type);
            return ret;
        }
    }

    return 0;
}
#endif //TINYNN_TESTUTIL_H