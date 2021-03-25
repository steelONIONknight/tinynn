//
// Created by lifan on 2021/2/26.
//

#include "cast.h"
#include <omp.h>

namespace tinynn
{

Cast::Cast()
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = true;
}

int Cast::load_param(const ParamDict &pd)
{
    type_from = pd.get(0, 0);
    type_to = pd.get(1, 0);

    return 0;
}

int Cast::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }
    int width = bottom_blob.width;
    int height = bottom_blob.height;
    int channel = bottom_blob.channel;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    size_t out_elemsize;

    if (type_to == 1)
    {
        //float32
        out_elemsize = 4 * elempack;
    }
    else if (type_to == 2)
    {
        //float16
        out_elemsize = 2 * elempack;
    }

    if (dims == 1)
    {
        top_blob.create(width, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        top_blob.create(width, height, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        top_blob.create(width, height, channel, out_elemsize, elempack, opt.blob_allocator);
    }

    if (top_blob.empty())
        return -100;

    size_t size = width * height * elempack;
    if (type_from == 1 && type_to == 2)
    {
#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < channel; ++i)
        {
            const float* ptr = bottom_blob.refer_channel(i);
            unsigned short* p = top_blob.refer_channel(i);
            for (int j = 0; j < size; ++j)
            {
                p[j] = float32_to_float16(ptr[j]);
            }
        }
    }
    else if (type_from == 2 && type_to == 1)
    {
#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < channel; ++i)
        {
            const unsigned short* ptr = bottom_blob.refer_channel(i);
            float* p = top_blob.refer_channel(i);
            for (int j = 0; j < size; ++j)
            {
                p[j] = float16_to_float32(ptr[j]);
            }
        }
    }

    return 0;
}

}