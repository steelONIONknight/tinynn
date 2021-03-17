//
// Created by lifan on 2021/3/14.
//

#ifndef TINYNN_TESTUTIL_H
#define TINYNN_TESTUTIL_H
#include "../cpu.h"
#include "../layer.h"
#include "../mat.h"
#include "prng.h"

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

#endif //TINYNN_TESTUTIL_H
