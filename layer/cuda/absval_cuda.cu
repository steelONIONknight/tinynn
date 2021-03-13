//
// Created by lifan on 2021/3/7.
//

//#if TINYNN_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../../cuda_util.h"

#include <iostream>

__global__ void cuda_absval_forward_inplace(float* input, const int input_size)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx > input_size)
        return;

    input[idx] = input[idx] >= 0 ? input[idx] : -input[idx];
}

namespace tinynn
{
int absval_forward_inplace(float* input, const int input_size)
{
    int block_x = ((input_size - 1) / 32 + 1 ) * 32;
    //dim3 block的x，y维不能超过1024
    block_x = block_x > 1024 ? 1024 : block_x;

    const dim3 block(block_x);
    const dim3 grid((input_size - 1) / block_x + 1);

    cuda_absval_forward_inplace<<<grid, block>>>(input, input_size);

    return 0;
}

}//namespace tinynn

//#endif