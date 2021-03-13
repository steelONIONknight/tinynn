//
// Created by lifan on 2021/2/25.
//

#include "absval.h"
#include <omp.h>
namespace tinynn
{
AbsVal::AbsVal()
{
    one_blob_only = true;
    support_inplace = true;
}

int AbsVal::forward_inplace(Mat &bottom_top_blob, const Option &opt) const
{
    int width = bottom_top_blob.width;
    int height = bottom_top_blob.height;
    int channel = bottom_top_blob.channel;
    int size = width * height;

#pragma omp parallel for num_threads(opt.num_thread)
    for (int i = 0; i < channel; ++i)
    {
        float* ptr = bottom_top_blob.refer_channel(i);

        for (int j = 0; j < size; ++j)
        {
            if (ptr[j] < 0)
                ptr[j] = -ptr[j];
        }
    }
    return 0;
}

}


