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

const int N_ITERATIONS = 32;

__global__ void cuda_innerproduct_forward(const float* input, const float* weight, float* output,
                                          const int w, const int h, const int c, const int num_output)
{
    unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int chaIdx = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int idx = rowIdx * w + colIdx + chaIdx * w * h;
    unsigned int output_idx;

    for (int p = 0; p < num_output; p++)
    {
        output_idx = idx + p * w * h * c;
        output[output_idx] = input[idx] * weight[output_idx];
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

__global__ void cuda_innerproduct_activation(const float* input, const float* bias_data, const float* activation_params,
                                             float* output, int activation_type, int num_output)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > num_output)
        return;

    output[idx] = input[idx] + bias_data[idx];

    if (activation_type == 1)
    {
        output[idx] = max(output[idx], 0.f);
    }
    else if (activation_type == 2)
    {
        float slope = activation_params[0];
        output[idx] = output[idx] > 0.f ? output[idx] : output[idx] * slope;
    }
    else if (activation_type == 3)
    {
        float min_val = activation_params[0];
        float max_val = activation_params[1];

        if (output[idx] < min_val)
            output[idx] = min_val;
        if (output[idx] > max_val)
            output[idx] = max_val;
    }
    else if (activation_type == 4)
    {
        output[idx] = (float)(1.f / 1.f + expf(-output[idx]));
    }
    else if (activation_type == 5)
    {
        output[idx] = (float)(output[idx] * tanhf(logf(expf(output[idx]) + 1.f)));
    }

}

__global__ void cuda_innerproduct_reduction_activation(const float* input, float* output, const float* bias_data,
                                                       const float* activation_params, int width, int activation_type)
{
    unsigned int laneIdx = threadIdx.x & 31;
    unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int rowIdx = N_ITERATIONS * blockIdx.y;

    unsigned int idx = rowIdx * width + colIdx;

    for (int i = 0; i < N_ITERATIONS; i++)
    {
        float v = idx < width * (i + 1) ? input[idx] : 0.f;

        for (int stride = 16; stride >= 1; stride /= 2)
            v += __shfl_down_sync(0xffffffff, v, stride);

        if (laneIdx == 0)
            atomicAdd(&output[rowIdx], v);

        __syncthreads();

        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            if (bias_data != nullptr)
                output[rowIdx] += bias_data[rowIdx];
            if (activation_type == 1)
            {
                output[rowIdx] = max(output[rowIdx], 0.f);
            }
            else if (activation_type == 2)
            {
                float slope = activation_params[0];
                output[rowIdx] = output[rowIdx] > 0.f ? output[rowIdx] : output[rowIdx] * slope;
            }
            else if (activation_type == 3)
            {
                float min_val = activation_params[0];
                float max_val = activation_params[1];

                if (output[rowIdx] < min_val)
                    output[rowIdx] = min_val;
                if (output[rowIdx] > max_val)
                    output[rowIdx] = max_val;
            }
            else if (activation_type == 4)
            {
                output[rowIdx] = (float)(1.f / 1.f + expf(-output[rowIdx]));
            }
            else if (activation_type == 5)
            {
                output[rowIdx] = (float)(output[rowIdx] * tanhf(logf(expf(output[rowIdx]) + 1.f)));
            }
        }
        rowIdx++;
        idx += width;
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
        block_x = ((bottom_blob.width - 1) / 32 + 1) * 32;
        block_x = block_x > 1024 ? 1024 : block_x;
        block_y = ((bottom_blob.height - 1) / 32 + 1) * 32;
        block_y = block_y > 1024 ? 1024 : block_y;

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
        block_x = ((bottom_blob.width - 1) / 32 + 1) * 32;
        block_x = block_x > 1024 ? 1024 : block_x;
        block_y = ((bottom_blob.height - 1) / 32 + 1) * 32;
        block_y = block_y > 1024 ? 1024 : block_y;

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
                                               bottom_blob.width, bottom_blob.height, bottom_blob.channel, num_output);
    cudaDeviceSynchronize();

    //**************just for debug***************************
//    float* h_intermediate_res;
//    h_intermediate_res = (float*)malloc(weight_data_size * sizeof(float));
//    cudaMemcpy(h_intermediate_res, intermediate_res, weight_data_size * sizeof(float), cudaMemcpyDeviceToHost);
//    for (int j = 0; j < weight.channel; j++)
//    {
//        setbuf(stdout, nullptr);
//        printf("channel: %d\n", j);
//        int index = j * weight.width * weight.height;
//        for (int i = 0; i < weight.width * weight.height; i++)
//        {
//            setbuf(stdout, nullptr);
//            printf("%f ", h_intermediate_res[index + i]);
//        }
//        setbuf(stdout, nullptr);
//        printf("\n");
//    }
//    free(h_intermediate_res);
    //**************just for debug***************************

    int sz = weight_data_size / num_output;
    block_x = ((sz - 1) / N_ITERATIONS + 1) * N_ITERATIONS;
    block_x = block_x > 1024 ? 1204 : block_x;
    grid_x = sz / N_ITERATIONS;
    grid_x = grid_x >= 1 ? grid_x : 1;
    grid_y = num_output / N_ITERATIONS;
    grid_y = grid_y >= 1 ? grid_y : 1;

    block.x = block_x;
    block.y = 1;
    block.z = 1;
    grid.x = grid_x;
    grid.y = grid_y;
    grid.z = 1;

    if (bias_term == 0)
    {
        cuda_innerproduct_reduction_activation<<<grid, block>>>(intermediate_res, (float*)top_blob.data,
                                                                nullptr, (float*)activation_params.data,
                                                                bottom_blob.width, activation_type);
    }
    else if (bias_term == 1)
    {
        cuda_innerproduct_reduction_activation<<<grid, block>>>(intermediate_res, (float*)top_blob.data,
                                                                (float*)bias_data.data, (float*)activation_params.data,
                                                                bottom_blob.width, activation_type);
    }
    cudaDeviceSynchronize();

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
//    cudaFree(dev_temp);
//    cudaFree(dev_sum);
//    free(temp);
//    free(sum);
    return 0;
}

} // namespace tinynn