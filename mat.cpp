//
// Created by lifan on 2021/1/26.
//

#include "mat.h"

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"

#include <cmath>

namespace tinynn
{
float float16_to_float32(unsigned short value)
{
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7C00) >> 10;
    unsigned short significand = value & 0x03FF;

    union
    {
        unsigned int u;
        float f;
    }temp;

    if (exponent == 0)
    {
        if (significand == 0)
        {
            //zero
            temp.u = sign << 31;
        }
        else
        {
            //not normalized
            exponent = 0;
            //规约化
            while ((significand & 0x200) == 0)
            {
                exponent++;
                significand <<= 1;
            }
            //IEEE 754 尾数隐藏了1
            significand <<= 1;
            significand &= 0x3FF;
            temp.u = ((sign << 31) | ((-exponent - 15 + 127) << 23) | (significand << 13));
        }
    }
    else if (exponent == 0x1F)
    {
        //inf or NaN
        temp.u = ((sign << 31) | (0xFF << 23) | (significand << 13));
    }
    else
    {
        //normalized
        temp.u = ((sign << 31) | ((exponent - 15 + 127) << 23) | (significand << 13));
    }
    return temp.f;
}
unsigned short float32_to_float16(float value)
{
    //float32
    //1: 8: 23
    union
    {
        unsigned int u;
        float f;
    }temp;
    temp.f = value;
    unsigned short sign = (temp.u & 0x80000000) >> 31;
    unsigned short exponent = (temp.u & 0x7F800000) >> 23;
    unsigned int significant = (temp.u & 0x007FFFFF);

    unsigned short fp16;

    if (exponent == 0)
    {
        //32bit浮点数的阶码的值为0，表示非常小的数字
        //16bit浮点数一定会溢出，应直接表示为0
        fp16 = (sign << 15) | (0x00 << 10) | (0x00);
    }
    else if (exponent == 0xFF)
    {
        //inf or NaN
        if (significant == 0)
        {
            //inf
            fp16 = (sign << 15) | (0x1F << 10) | (0x00);
        }
        else
        {
            //NaN
            fp16 = (sign << 15) | (0x1F << 10) | (0x200);
        }
    }
    else
    {
        //normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            //overflow, inf
            fp16 = (sign << 15) | (0x1F << 10) | (0x00);
        }
        else if (newexp <= 0)
        {
            if (newexp >= -10)
            {
                //very small fp16
                unsigned short newsig = (significant | 0x80000) >> (14 - newexp);
                fp16 = (sign << 15) | (0x00 << 10) | newsig;
            }
            else
            {
                //0
                fp16 = (sign << 15) | (0x00 << 10) | (0x00);
            }
        }
        else
        {
            //normal
            fp16 = (sign << 15) | (newexp << 10) | (significant >> 13);
        }
    }
    return fp16;
}

Mat Mat::from_float16(const unsigned short *data, int size) {
    Mat m(size);
    if (m.empty())
        return m;

    float *ptr = m;

    int remain = size;
    for (; remain > 0; remain--)
    {
        *ptr = float16_to_float32(*data);
        data++;
        ptr++;
    }
    return m;
}
void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt)
{
    Layer* padding = create_layer(LayerType::Padding);

    ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);

    padding->load_param(pd);

    padding->create_pipeline(opt);

    padding->forward(src, dst, opt);

    padding->destroy_pipeline(opt);

    delete padding;
}

} // namespace tinynn