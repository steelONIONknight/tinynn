//
// Created by lifan on 2021/5/31.
//

#include "test_padding.h"
static int test(const tinynn::Mat& a, int top, int bottom, int left, int right, int front, int behind, int type, float value, int per_channel_pad_data_size)
{
    tinynn::ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, value);
    pd.set(6, per_channel_pad_data_size);
    pd.set(7, front);
    pd.set(8, behind);

    std::vector<tinynn::Mat> weights(per_channel_pad_data_size ? 1 : 0);
    if (per_channel_pad_data_size)
        weights[0] = RandomMat(per_channel_pad_data_size);

    int ret = test_layer<tinynn::Padding>("Padding", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_padding failed a.dims=%d a=(%d %d %d) top=%d bottom=%d left=%d right=%d front=%d behind=%d type=%d value=%f per_channel_pad_data_size=%d\n", a.dims, a.w, a.h, a.c, top, bottom, left, right, front, behind, type, value, per_channel_pad_data_size);
    }

    return ret;
}

static int test_padding_0()
{
    tinynn::Mat a = RandomMat(9, 11, 24);
    tinynn::Mat b = RandomMat(10, 13, 12);
    tinynn::Mat c = RandomMat(8, 9, 13);

    return 0
           || test(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test(a, 2, 2, 2, 2, 0, 0, 0, 1.f, 0)
           || test(b, 2, 2, 2, 2, 0, 0, 0, 2.f, 0)
           || test(c, 2, 2, 2, 2, 0, 0, 0, -3.f, 0)

           || test(a, 2, 1, 2, 1, 0, 0, 0, 0.f, a.channel)
           || test(b, 2, 1, 2, 1, 0, 0, 0, 0.f, b.channel)
           || test(c, 2, 1, 2, 1, 0, 0, 0, 0.f, c.channel)

           || test(a, 0, 1, 0, 1, 0, 0, 1, 0.f, 0)
           || test(b, 0, 1, 0, 1, 0, 0, 1, 0.f, 0)
           || test(c, 0, 1, 0, 1, 0, 0, 1, 0.f, 0)

           || test(a, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)
           || test(b, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)
           || test(c, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)

           || test(a, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test(b, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test(c, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)

           || test(a, 1, 1, 1, 1, 1, 1, 0, -1.f, 0)
           || test(b, 1, 1, 1, 1, 1, 1, 0, -2.f, 0)
           || test(c, 1, 1, 1, 1, 1, 1, 0, 3.f, 0)

           || test(a, 2, 1, 0, 0, 2, 3, 0, 0.f, a.channel + 5)
           || test(b, 2, 1, 0, 0, 2, 3, 0, 0.f, b.channel + 5)
           || test(c, 2, 1, 0, 0, 2, 3, 0, 0.f, c.channel + 5)

           || test(a, 1, 2, 3, 4, 8, 4, 0, 0.f, a.channel + 12)
           || test(b, 1, 2, 3, 4, 8, 4, 0, 0.f, b.channel + 12)
           || test(c, 1, 2, 3, 4, 8, 4, 0, 0.f, c.channel + 12)

           || test(a, 0, 0, 0, 0, 3, 1, 1, 0.f, 0)
           || test(b, 0, 0, 0, 0, 3, 1, 1, 0.f, 0)
           || test(c, 0, 0, 0, 0, 3, 1, 1, 0.f, 0)

           || test(a, 2, 0, 1, 0, 4, 4, 1, 0.f, 0)
           || test(b, 2, 0, 1, 0, 4, 4, 1, 0.f, 0)
           || test(c, 2, 0, 1, 0, 4, 4, 1, 0.f, 0)

           || test(a, 2, 0, 2, 0, 0, 2, 2, 0.f, 0)
           || test(b, 2, 0, 2, 0, 0, 2, 2, 0.f, 0)
           || test(c, 2, 0, 2, 0, 0, 2, 2, 0.f, 0)

           || test(a, 4, 2, 1, 3, 3, 5, 2, 0.f, 0)
           || test(b, 4, 2, 1, 3, 3, 5, 2, 0.f, 0)
           || test(c, 4, 2, 1, 3, 3, 5, 2, 0.f, 0);
}

static int test_padding_1()
{
    tinynn::Mat a = RandomMat(15, 24);
    tinynn::Mat b = RandomMat(19, 12);
    tinynn::Mat c = RandomMat(17, 15);

    return 0
           || test(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test(a, 0, 0, 1, 1, 0, 0, 0, 1.f, 0)
           || test(b, 0, 0, 1, 1, 0, 0, 0, 2.f, 0)
           || test(c, 0, 0, 1, 1, 0, 0, 0, -3.f, 0)

           || test(a, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)
           || test(b, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)
           || test(c, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)

           || test(a, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)
           || test(b, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)
           || test(c, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)

           || test(a, 2, 2, 2, 2, 0, 0, 0, 1.f, 0)
           || test(b, 2, 2, 2, 2, 0, 0, 0, 2.f, 0)
           || test(c, 2, 2, 2, 2, 0, 0, 0, -3.f, 0)

           || test(a, 8, 8, 2, 5, 0, 0, 0, -1.f, 0)
           || test(b, 8, 8, 2, 5, 0, 0, 0, -2.f, 0)
           || test(c, 8, 8, 2, 5, 0, 0, 0, 3.f, 0)

           || test(a, 3, 1, 3, 1, 0, 0, 1, 0.f, 0)
           || test(b, 3, 1, 3, 1, 0, 0, 1, 0.f, 0)
           || test(c, 3, 1, 3, 1, 0, 0, 1, 0.f, 0)

           || test(a, 4, 4, 0, 1, 0, 0, 1, 0.f, 0)
           || test(b, 4, 4, 0, 1, 0, 0, 1, 0.f, 0)
           || test(c, 4, 4, 0, 1, 0, 0, 1, 0.f, 0)

           || test(a, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test(b, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test(c, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)

           || test(a, 2, 6, 1, 0, 0, 0, 2, 0.f, 0)
           || test(b, 2, 6, 1, 0, 0, 0, 2, 0.f, 0)
           || test(c, 2, 6, 1, 0, 0, 0, 2, 0.f, 0);
}

static int test_padding_2()
{
    tinynn::Mat a = RandomMat(128);
    tinynn::Mat b = RandomMat(124);
    tinynn::Mat c = RandomMat(127);

    return 0
           || test(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test(a, 0, 0, 2, 2, 0, 0, 0, 1.f, 0)
           || test(b, 0, 0, 2, 2, 0, 0, 0, 2.f, 0)
           || test(c, 0, 0, 2, 2, 0, 0, 0, -3.f, 0)

           || test(a, 0, 0, 16, 8, 0, 0, 0, -1.f, 0)
           || test(b, 0, 0, 16, 8, 0, 0, 0, -2.f, 0)
           || test(c, 0, 0, 16, 8, 0, 0, 0, 3.f, 0)

           || test(a, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)
           || test(b, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)
           || test(c, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)

           || test(a, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)
           || test(b, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)
           || test(c, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)

           || test(a, 0, 0, 2, 3, 0, 0, 2, 0.f, 0)
           || test(b, 0, 0, 2, 3, 0, 0, 2, 0.f, 0)
           || test(c, 0, 0, 2, 3, 0, 0, 2, 0.f, 0)

           || test(a, 0, 0, 10, 6, 0, 0, 2, 0.f, 0)
           || test(b, 0, 0, 10, 6, 0, 0, 2, 0.f, 0)
           || test(c, 0, 0, 10, 6, 0, 0, 2, 0.f, 0);
}

int test_padding()
{
    SRAND(7767517);

    return test_padding_0() || test_padding_1() || test_padding_2();
}