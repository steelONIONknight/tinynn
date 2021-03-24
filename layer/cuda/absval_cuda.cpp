//
// Created by lifan on 2021/2/28.
//

#include "absval_cuda.h"

namespace tinynn
{
int absval_forward_inplace(float* input, const int input_size);
AbsVal_cuda::AbsVal_cuda()
{
    one_blob_only = true;
    support_inplace = true;
    support_cuda = true;
}

int AbsVal_cuda::forward_inplace(CudaMat &bottom_top_blob, const Option &/*opt*/) const
{
    const int total_sz = bottom_top_blob.total();

    absval_forward_inplace((float*)bottom_top_blob.data, total_sz);
    return 0;
}

}

