//
// Created by lifan on 2021/4/10.
//

#include "test_innerproduct.h"

static int test(const tinynn::Mat& a, int outch, int bias)
{
    //set parameters of the layer
    tinynn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, bias);
    pd.set(2, outch * a.width * a.height * a.channel);

//    int activation_type = RAND() % 6;
    //激活函数暂时先设置为relu
    int activation_type = 1;
    tinynn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); //alpha
    activation_params[1] = RandomFloat(0, 1);  //beta

    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<tinynn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * a.width * a.height * a.channel);
    if (bias)
        weights[1] = RandomMat(outch * a.width * a.height * a.channel);

    //**********just for debug**********************
//    for (int j = 0; j < a.channel; ++j)
//    {
//        setbuf(stdout, nullptr);
//        printf("channel %d\n", j);
//        int index = j * a.cstep;
//        for (int i = 0; i < a.width * a.height; ++i)
//        {
//            setbuf(stdout, nullptr);
//            printf("%f, ", a[index + i]);
//            if ((i + 1) % a.width == 0)
//            {
//                setbuf(stdout, nullptr);
//                printf("\n");
//            }
//        }
//    }
//
//    for (int j = 0; j < weights[0].channel; ++j)
//    {
//        setbuf(stdout, nullptr);
//        printf("channel %d\n", j);
//        int index = j * weights[0].cstep;
//        for (int i = 0; i < weights[0].width * weights[0].height; ++i)
//        {
//            setbuf(stdout, nullptr);
//            printf("%f, ", weights[0][index + i]);
//            if ((i + 1) % weights[0].width == 0)
//            {
//                setbuf(stdout, nullptr);
//                printf("\n");
//            }
//        }
//    }
//    printf("\n");
    //**********just for debug**********************

    int ret = test_layer<tinynn::InnerProduct>("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct failed a.dims=%d a=(%d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n",
                a.dims, a.width, a.height, a.channel, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_0()
{
    return 0
            || test(RandomMat(1, 3, 1), 1, 1)
            || test(RandomMat(3, 2, 2), 2, 1)
            || test(RandomMat(3, 2, 2), 7, 1)
            || test(RandomMat(9, 3, 8), 7, 1)
            || test(RandomMat(2, 2, 8), 8, 1)
            || test(RandomMat(4, 3, 15), 8, 1)
            || test(RandomMat(6, 2, 16), 16, 1)
            || test(RandomMat(6, 2, 16), 7, 1)
            || test(RandomMat(6, 2, 5), 16, 1)
            || test(RandomMat(256, 4, 2), 128, 1)
            || test(RandomMat(128, 512, 3), 16, 1)
            || test(RandomMat(129, 513, 3), 32, 1);
}

static int test_innerproduct_1()
{
    return 0
            || test(RandomMat(1, 1), 1, 1)
            || test(RandomMat(3, 2), 2, 1)
            || test(RandomMat(9, 8), 7, 1)
            || test(RandomMat(2, 8), 8, 1)
            || test(RandomMat(4, 15), 8, 1)
            || test(RandomMat(6, 16), 16, 1)
            || test(RandomMat(6, 16), 7, 1)
            || test(RandomMat(512, 4), 127, 1)
            || test(RandomMat(6, 5), 17, 1);
}

static int test_innerproduct_2()
{
    return 0
            || test(RandomMat(1), 1, 1)
            || test(RandomMat(2), 2, 1)
            || test(RandomMat(8), 7, 1)
            || test(RandomMat(8), 8, 1)
            || test(RandomMat(15), 8, 1)
            || test(RandomMat(16), 16, 1)
            || test(RandomMat(16), 7, 1)
            || test(RandomMat(5), 16, 1)
            || test(RandomMat(32), 16, 1)
            || test(RandomMat(12), 16, 1)
            || test(RandomMat(16), 12, 1)
            || test(RandomMat(65), 8, 1)
            || test(RandomMat(65), 16, 1)
            || test(RandomMat(1230), 10, 1)
            || test(RandomMat(5630), 19, 1)
            || test(RandomMat(24), 32, 1);

}

int test_innerproduct()
{
    SRAND(7767517);

    return 0
            || test_innerproduct_0()
            || test_innerproduct_1()
            || test_innerproduct_2();
}