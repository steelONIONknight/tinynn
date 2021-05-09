//
// Created by lifan on 2021/3/26.
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../../cuda_util.h"

#include <iostream>

#include "../../mat.h"
#include "../innerproduct.h"

#define FULL_MASK 0xffffffff
const int N_ITERATIONS = 32;

//parameter pitch_num is in number of elements, not in bytes!
__global__ void cuda_innerproduct_forward(const float* input, const float* weight, float* output, const int w,
                                          const int h, const int c, const int num_output, size_t pitch_num)
{
    unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int chaIdx = blockIdx.z * blockDim.z + threadIdx.z;

    if (c == 1)
        pitch_num = w * h;

    unsigned int idx = rowIdx * w + colIdx;
    unsigned int input_idx = idx + chaIdx * pitch_num;
    unsigned int output_idx = idx + chaIdx * w * h;

    if (rowIdx >= h || colIdx >= w)
        return;

    //有些线程用不上
    if (input_idx >= pitch_num * c)
        return;

    for (int p = 0; p < num_output; p++)
    {

        output[output_idx] = input[input_idx] * weight[output_idx];
//        printf("%d ", input_idx);
//        __syncthreads();
//        printf("%f,", input[input_idx]);
//        __syncthreads();
//        printf("%d ", output_idx);
//        __syncthreads();
//        printf("%f,", output[output_idx]);
//        __syncthreads();

        output_idx += w * h * c;
    }

}

__global__ void cuda_innerproduct_reduction(float* input, float* output, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    //unroll loop 8
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    float* i_data = input + blockIdx.x * blockDim.x * 8;

    if (tid + blockDim.x * 7 < n)
    {
        float a1 = input[idx + 0 * blockDim.x];
        float a2 = input[idx + 1 * blockDim.x];
        float a3 = input[idx + 2 * blockDim.x];
        float a4 = input[idx + 3 * blockDim.x];
        float b1 = input[idx + 4 * blockDim.x];
        float b2 = input[idx + 5 * blockDim.x];
        float b3 = input[idx + 6 * blockDim.x];
        float b4 = input[idx + 7 * blockDim.x];
        input[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;

        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride > 32; stride /= 2)
    {
        if (tid < stride)
        {
            i_data[tid] += i_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile float* vmem = i_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
    {
        output[blockIdx.x] = i_data[0];
    }
}

__global__ void cuda_innerproduct_activation(float* input, const float* bias_data, const float* activation_params,
                                             int activation_type, int num_output)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > num_output)
        return;

    input[idx] += bias_data[idx];

    if (activation_type == 1)
    {
        input[idx] = max(input[idx], 0.f);
    }
    else if (activation_type == 2)
    {
        float slope = activation_params[0];
        input[idx] = input[idx] > 0.f ? input[idx] : input[idx] * slope;
    }
    else if (activation_type == 3)
    {
        float min_val = activation_params[0];
        float max_val = activation_params[1];

        if (input[idx] < min_val)
            input[idx] = min_val;
        if (input[idx] > max_val)
            input[idx] = max_val;
    }
    else if (activation_type == 4)
    {
        input[idx] = (float)(1.f / 1.f + expf(-input[idx]));
    }
    else if (activation_type == 5)
    {
        input[idx] = (float)(input[idx] * tanhf(logf(expf(input[idx]) + 1.f)));
    }

}

__global__ void cuda_innerproduct_reduction(const float* input, float* output, int width, const int weight_data_size,
                                            const int num_output)
{
    unsigned int laneIdx = threadIdx.x & 31;
    unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int rowIdx = N_ITERATIONS * blockIdx.y;

    unsigned int idx = rowIdx * width + colIdx;

    if (idx >= weight_data_size)
        return;

    for (int i = 0; i < N_ITERATIONS; i++)
    {
        if (rowIdx >= num_output)
            return;

        float v = 0.f;
        const unsigned int mask = __ballot_sync(FULL_MASK, idx < width * (rowIdx + 1));
        if (idx < width * (rowIdx + 1))
        {
            v = input[idx];
            for (int stride = warpSize / 2; stride > 0; stride /= 2)
                v += __shfl_down_sync(mask, v, stride);
        }
//        float v = idx < width * (rowIdx + 1) ? input[idx] : 0.f;
//        for (int stride = warpSize / 2; stride >= 1; stride /= 2)
//            v += __shfl_down_sync(FULL_MASK, v, stride);

        if (laneIdx == 0)
            atomicAdd(&output[rowIdx], v);

        rowIdx++;
        idx += width;
    }
}
__global__ void watch_gpu_mem(const float* input, const int w, const int h, const int c, const size_t pitch)
{
    unsigned int idx;
    printf("%d\n", pitch);
    unsigned int pitch_num = pitch / sizeof(float);
//    for (int q = 0; q < c; q++)
//    {
//        for (int i = 0; i < w * h; i++)
//        {
//            idx = i;
//            float temp = *(float*)((char*)(input + idx) + q * pitch);
//            printf("%f ", temp);
//        }
//    }
//    printf("\n");
    for (int q = 0; q < c; q++)
    {
        for (int i = 0; i < w * h; i++)
        {
//            printf("%f ", input[i + q * pitch_num]);
            printf("%d ", i + q * pitch_num);
        }
    }
}
namespace tinynn
{
int innerproduct_forward(const CudaMat& bottom_blob, const CudaMat& weight, const CudaMat& activation_params,
                         const CudaMat& bias_data, CudaMat& top_blob, int num_output, int bias_term, int weight_data_size,
                         int activation_type)
{
    dim3 block;
    dim3 grid;
    int dims = bottom_blob.dims;
    int block_x = 1, block_y = 1, block_z = 1;
    int grid_x = 1, grid_y = 1, grid_z = 1;
    if (dims == 1)
    {
        block_x = ((bottom_blob.width - 1) / 32 + 1) * 32;
        block_x = block_x > 1024 ? 1024 : block_x;

        grid_x = (bottom_blob.width - 1) / block_x + 1;

        block.x = block_x;
        block.y = block_y;
        block.z = block_z;
        grid.x = grid_x;
        grid.y = grid_y;
        grid.z = grid_z;
    }
    else if (dims == 2)
    {
        //每个block内最大线程的数量暂时设置为512
        block_x = ((bottom_blob.width - 1) / 32 + 1) * 32;
        block_x = block_x > 64 ? 64 : block_x;
        block_y = ((bottom_blob.height - 1) / 8 + 1) * 8;
        block_y = block_y > 8 ? 8 : block_y;
        grid_x = (bottom_blob.width - 1) / block_x + 1;
        grid_y = (bottom_blob.height - 1) / block_y + 1;

        block.x = block_x;
        block.y = block_y;
        block.z = block_z;
        grid.x = grid_x;
        grid.y = grid_y;
        grid.z = grid_z;
    }
    else if (dims == 3)
    {
        //每个block内最大线程的数量暂时设置为512
//        block_x = ((bottom_blob.width - 1) / 32 + 1) * 32;
//        block_x = block_x > 1024 ? 1024 : block_x;
        block_x = ((bottom_blob.width - 1) / 32 + 1) * 32;
        block_x = block_x > 64 ? 64 : block_x;
        block_y = ((bottom_blob.height - 1) / 8 + 1) * 8;
        block_y = block_y > 8 ? 8 : block_y;
        block_z = 1;

        grid_x = (bottom_blob.width - 1) / block_x + 1;
        grid_y = (bottom_blob.height - 1) / block_y + 1;
        grid_z = (bottom_blob.channel - 1) /block_z + 1;

        block.x = block_x;
        block.y = block_y;
        block.z = block_z;
        grid.x = grid_x;
        grid.y = grid_y;
        grid.z = grid_z;
    }

    float* intermediate_res;
    cudaMalloc((void**)& intermediate_res, weight_data_size * sizeof(float));

    cuda_innerproduct_forward<<<grid, block>>>((float*)bottom_blob.data, (float*)weight.data, intermediate_res,
                                               bottom_blob.width, bottom_blob.height, bottom_blob.channel, num_output,
                                               bottom_blob.pitch / sizeof(float));
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();

    //**************just for debug***************************
//    watch_gpu_mem<<<1, 1>>>((float*)bottom_blob.data, bottom_blob.width, bottom_blob.height, bottom_blob.channel,
//                            bottom_blob.pitch);
//    cudaDeviceSynchronize();

//    float* h_intermediate_res;
//    h_intermediate_res = (float*)malloc(weight_data_size * sizeof(float));
//    cudaMemcpy(h_intermediate_res, intermediate_res, weight_data_size * sizeof(float), cudaMemcpyDeviceToHost);
//
//    for (int i = 0; i < weight_data_size; i++)
//    {
//        setbuf(stdout, nullptr);
//        printf("%f, ", h_intermediate_res[i]);
//    }
//    setbuf(stdout, nullptr);
//    printf("\n");
//
//    free(h_intermediate_res);
    //**************just for debug***************************

    //grid和block的配置有问题,需要调整
    int sz = weight_data_size / num_output;
    block_x = ((sz - 1) / N_ITERATIONS + 1) * N_ITERATIONS;
    block_x = block_x > 1024 ? 1024 : block_x;
//    grid_x = sz / N_ITERATIONS;
//    grid_x = grid_x >= 1 ? grid_x : 1;
//    grid_y = num_output / N_ITERATIONS;
//    grid_y = grid_y >= 1 ? grid_y : 1;
    grid_x = (sz - 1) / block_x + 1;
    grid_y = (num_output - 1) / N_ITERATIONS + 1;

    block.x = block_x;
    block.y = 1;
    block.z = 1;
    grid.x = grid_x;
    grid.y = grid_y;
    grid.z = 1;

    if (bias_term == 0)
    {
        cuda_innerproduct_reduction<<<grid, block>>>(intermediate_res, (float*)top_blob.data,
                                                                bottom_blob.width * bottom_blob.height * bottom_blob.channel,
                                                                weight_data_size, num_output);
    }
    else if (bias_term == 1)
    {
        cuda_innerproduct_reduction<<<grid, block>>>(intermediate_res, (float*)top_blob.data,
                                                                bottom_blob.width * bottom_blob.height * bottom_blob.channel,
                                                                weight_data_size, num_output);
    }
    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    //**************just for debug***************************
//    float* h_top_blob;
//    h_top_blob = (float*)malloc(num_output * sizeof(float));
//    cudaMemcpy(h_top_blob, top_blob.data, num_output * sizeof(float), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < num_output; i++)
//    {
//        printf("%f ", h_top_blob[i]);
//    }
//    printf("\n");
//    free(h_top_blob);
    //**************just for debug***************************

    block_x = (num_output - 1) / 32 + 1;
    block_x = block_x > 1024 ? 1024 : block_x;

    grid_x = (num_output - 1) / block_x + 1;

    block.x = block_x;
    block.y = 1;
    block.z = 1;
    grid.x = grid_x;
    grid.y = 1;
    grid.z = 1;

    cuda_innerproduct_activation<<<grid, block>>>((float*)top_blob.data, (float*)bias_data.data, (float*)activation_params.data,
                                                  activation_type, num_output);

    //    float* dev_temp;
//    cudaMalloc((void**)& dev_temp, grid.x * sizeof(float) * num_output);
//    //gpu内存初始化
//    cudaMemset(dev_temp, 0, grid.x * sizeof(float) * num_output);
//    float* temp;
//    temp = (float*)malloc(grid.x * sizeof(float) * num_output);
//
//    cudaDeviceSynchronize();
//    cudaMemcpy(temp, dev_temp, grid.x * sizeof(float) * num_output, cudaMemcpyDeviceToHost);


//    float* sum;
//    sum = (float*)malloc(num_output * sizeof(float));
//    std::fill_n(sum, num_output, 0.f);
//
//    for (int i = 0; i < num_output; i++)
//    {
//        for (int j = 0; j < grid.x; j++)
//        {
//            sum[i] += temp[i * grid.x + j];
//        }
//    }

//    float* dev_sum;
//    cudaMalloc((void**)& dev_sum, num_output * sizeof(float));
//    cudaMemcpy(dev_sum, sum, num_output * sizeof(float), cudaMemcpyHostToDevice);
//
//    block_x = ((num_output - 1) / 32 + 1) * 32;
//    block_x = block_x > 1024 ? 1024 : block_x;
//    grid_x = (num_output - 1) / block_x + 1;
//
//    block.x = block_x;
//    block.y = 1;
//    block.z = 1;
//    grid.x = grid_x;
//    grid.y = 1;
//    grid.z = 1;
//
//    cuda_innerproduct_activation<<<grid, block>>>(dev_sum, (float*)bias_data.data, (float*)activation_params.data,
//                                                  (float*)top_blob.data, activation_type, num_output);
//    cudaDeviceSynchronize();

    cudaFree(intermediate_res);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
//    cudaFree(dev_temp);
//    cudaFree(dev_sum);
//    free(temp);
//    free(sum);
    return 0;
}

} // namespace tinynn