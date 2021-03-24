//
// Created by lifan on 2021/3/21.
//
#include "test_absval.h"

static int test(const tinynn::Mat& a)
{
    tinynn::ParamDict pd;
    std::vector<tinynn::Mat> weights(0);
    int ret = test_layer<tinynn::AbsVal>("AbsVal", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_absval failed a.dims = %d a = (%d %d %d)\n", a.dims, a.width, a.height, a.channel);
    }
    return ret;
}

static int test_absval_0()
{
    return 0 || test(RandomMat(5, 7, 24))
             || test(RandomMat(7, 9, 12))
             || test(RandomMat(3, 5, 13));
}

static int test_absval_1()
{
    return 0 || test(RandomMat(15, 24))
             || test(RandomMat(19, 12))
             || test(RandomMat(17, 15));
}

static int test_absval_2()
{
    return 0 || test(RandomMat(124))
             || test(RandomMat(127))
             || test(RandomMat(128));
}

int test_absval()
{
    SRAND(7767517);
    return 0 || test_absval_0()
             || test_absval_1()
             || test_absval_2();

}