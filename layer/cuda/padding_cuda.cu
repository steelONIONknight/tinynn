//
// Created by lifan on 2021/5/17.
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../../cuda_util.h"

#include <iostream>

#include "../../mat.h"
#include "../padding.h"

template<typename T>
__global__ void cuda_copy_make_border_image_type0(const T* src, T* dst, int top, int left, T v, int src_w, int src_h,
                                                  int dst_w, int dst_h)
{
    unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx >= dst_h || colIdx >= dst_w)
        return;

    unsigned int idx = rowIdx * dst_w + colIdx;
    unsigned int src_idx = (rowIdx - top) * src_w + colIdx - left;
    T output_value = v;

    if ((rowIdx >= top && rowIdx < top + src_h) && (colIdx >= left && colIdx < left + src_w))
    {
        output_value = src[src_idx];
    }
    dst[idx] = output_value;
}

template<typename T>
__global__ void cuda_copy_make_border_image_type1(const T* src, T* dst, int top, int left, int src_w, int src_h,
                                                  int dst_w, int dst_h)
{
    unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx >= dst_h || colIdx >= dst_w)
        return;

    unsigned int idx = rowIdx * dst_w + colIdx;
    unsigned int src_idx = (rowIdx - top) * src_w + colIdx - left;
    T output_value = 0;

    if (rowIdx < top)
    {
        if (colIdx < left)
            output_value = src[0];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = src[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = src[src_w - 1];
    }
    else if (rowIdx >= top && rowIdx < top + src_h)
    {
        if (colIdx < left)
            output_value = src[(rowIdx - top) * src_w];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = src[src_idx];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = src[(rowIdx - top) * src + src_w - 1];
    }
    else if (rowIdx >= top + src_h && rowIdx < dst_h)
    {
        if (colIdx < left)
            output_value = src[(src_h - 1) * src_w];
        else if (colIdx >= left && colIdx < left +src_w)
            output_value = src[(src_h - 1) * src_w + colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = src[src_w * src_h - 1];
    }
    dst[idx] = output_value;

}

template<typename T>
__global__ void cuda_copy_make_border_image_type2(const T* src, T* dst, int top, int left, int src_w, int src_h,
                                                  int dst_w, int dst_h)
{
    unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx >= dst_h || colIdx >= dst_w)
        return;

    unsigned int idx = rowIdx * dst_w + colIdx;

    T output_value = 0;

    if (rowIdx < top)
    {
        const T* inptr = src + (top - rowIdx) * src_w;
        if (colIdx < left)
            output_value = inptr[left - colIdx];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = inptr[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = inptr[src_w - (colIdx - left - src_w) - 2];
    }
    else if (rowIdx >= top && rowIdx < top + src_h)
    {
        const T* inptr = src + (rowIdx - top) * src_w;
        if (colIdx < left)
            output_value = inptr[left - colIdx];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = inptr[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = inptr[src_w - (colIdx - left - src_w) - 2];
    }
    else if (rowIdx >= top + src_h && rowIdx < dst_h)
    {
        const T* inptr = src + (src_h - (rowIdx - top - src_h) - 2) * src_w;
        if (colIdx < left)
            output_value = inptr[left - colIdx];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = inptr[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = inptr[src_w - (colIdx - left - src_w) - 2];
    }
    dst[idx] = output_value;
}

template<typename T>
__global__ void cuda_copy_make_border_image_3d_type0(const T* src, T* dst, int front, int top, int left, T v,
                                                     const T* per_channel_pad_data, int per_channel_pad_data_size,
                                                     int src_w, int src_h, int src_c, int dst_w, int dst_h, int dst_c,
                                                     size_t pitch_num)
{
    unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int chaIdx = blockIdx.z * blockDim.z + threadIdx.z;

    if (rowIdx >= dst_h || colIdx >= dst_w || chaIdx >= dst_c)
        return;

    T* output_ptr = dst + rowIdx * dst_w + colIdx + pitch_num * chaIdx;

    T padding_value;
    T output_value;

    padding_value = per_channel_pad_data_size ? per_channel_pad_data[chaIdx] : v;

    output_value = padding_value;

    if (chaIdx < top || chaIdx >= top + src_c)
    {

    }
    else if ((rowIdx >= top && rowIdx < top + src_h) &&
            (colIdx >= left && colIdx < left + src_w))
    {
        output_value = src[(rowIdx - top) * src_w + colIdx - left + pitch_num * (chaIdx - front)];
    }

    *output_ptr = output_value;
}

template<typename T>
__global__ void cuda_copy_make_border_image_3d_type1(const T* src, T* dst, int front, int top, int left, T v,
                                                     const T* per_channel_pad_data, int per_channel_pad_data_size,
                                                     int src_w, int src_h, int src_c, int dst_w, int dst_h, int dst_c,
                                                     size_t pitch_num)
{
    unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int chaIdx = blockIdx.z * blockDim.z + threadIdx.z;

    if (rowIdx >= dst_h || colIdx >= dst_w || chaIdx >= dst_c)
        return;

    int q = (int)chaIdx - front;
    q = q < 0 ? 0 : q;
    q = q >= src_c - 1 ? src_c - 1 : q;

    const size_t dst_channel_step = pitch_num * chaIdx;
    const size_t src_channel_step = pitch_num * q;

    T* output_ptr = dst + rowIdx * dst_w + colIdx + dst_channel_step;

    T padding_value;
    T output_value;

    padding_value = per_channel_pad_data_size ? per_channel_pad_data[chaIdx] : v;
    output_value = padding_value;

    if (rowIdx < top)
    {
        const T* input_ptr = src + src_channel_step;
        if (colIdx < left)
            output_value = input_ptr[0];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = input_ptr[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = input_ptr[src_w - 1];
    }
    else if (rowIdx >= top && rowIdx < top + src_h)
    {
        const T* input_ptr = src + src_channel_step + (rowIdx - top) * src_w;
        if (colIdx < left)
            output_value = input_ptr[0];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = input_ptr[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = input_ptr[src_w - 1];
    }
    else if (rowIdx >= top + src_h && rowIdx < dst_h)
    {
        const T* input_ptr = src + src_channel_step;
        if (colIdx < left)
            output_value = input_ptr[0];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = input_ptr[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = input_ptr[src_w - 1];
    }
    *output_ptr = output_value;
}

template<typename T>
__global__ void cuda_copy_make_border_image_3d_type2(const T* src, T* dst, int front, int top, int left, T v,
                                                     const T* per_channel_pad_data, int per_channel_pad_data_size,
                                                     int src_w, int src_h, int src_c, int dst_w, int dst_h, int dst_c,
                                                     size_t pitch_num)
{
    unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int chaIdx = blockIdx.z * blockDim.z + threadIdx.z;

    if (rowIdx >= dst_h || colIdx >= dst_w || chaIdx >= dst_c)
        return;

    int q = (int)colIdx - front;
    q = abs(q);
    q = src_c - 1 - abs(q - (src_c - 1));

    const size_t dst_channel_step = pitch_num * chaIdx;
    const size_t src_channel_step = pitch_num * q;

    T* output_ptr = dst + rowIdx * dst_w + colIdx + dst_channel_step;

    T padding_value;
    T output_value;

    padding_value = per_channel_pad_data_size ? per_channel_pad_data[chaIdx] : v;
    output_value = padding_value;

    if (rowIdx < top)
    {
        const T* input_ptr = src + src_channel_step + (top - rowIdx) * src_w;
        if (colIdx < left)
            output_value = input_ptr[left - colIdx];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = input_ptr[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = input_ptr[src_w - (colIdx - left - src_w) - 2];
    }
    else if (rowIdx >= top && rowIdx < top + src_h)
    {
        const T* input_ptr = src + src_channel_step + (rowIdx - top) * src_w;
        if (colIdx < left)
            output_value = input_ptr[left - colIdx];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = input_ptr[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = input_ptr[src_w - (colIdx - left - src_w) - 2];
    }
    else if (rowIdx >= top + src_h && rowIdx < dst_h)
    {
        const T* input_ptr = src + (src_h - (rowIdx - top - src_h) - 2) * src_w;
        if (colIdx < left)
            output_value = input_ptr[left - colIdx];
        else if (colIdx >= left && colIdx < left + src_w)
            output_value = input_ptr[colIdx - left];
        else if (colIdx >= left + src_w && colIdx < dst_w)
            output_value = input_ptr[src_w - (colIdx - left - src_w) - 2];
    }

    *output_ptr = output_value;
}

namespace tinynn
{
template<typename T>
int copy_make_border_image_cuda(const CudaMat& bottom_blob, CudaMat& top_blob, int top, int left, int type, T v)
{
    dim3 block;
    dim3 grid;
    int block_x = 1, block_y = 1, block_z = 1;
    int grid_x = 1, grid_y = 1, grid_z = 1;
    
    int w = top_blob.width;
    int h = top_blob.height;
    int channels = top_blob.channel;
    size_t elemsize = top_blob.elemsize;

    block_x = w;
    if (block_x > 32) block_x = 32;
    block_y = h;
    if (block_y > 8) block_y = 8;

    grid_x = (w - 1) / block_x + 1;
    grid_y = (h - 1) / block_y + 1;
    grid_z = (channels - 1) / block_z + 1;

    block.x = block_x;
    block.y = block_y;
    block.z = block_z;
    grid.x = grid_x;
    grid.y = grid_y;
    grid.z = grid_z;

    if (type == 0)
    {
        if (elemsize == 1)
        {
            //TODO
//            cuda_copy_make_border_image_type0<signed char><<<grid, block>>>(static_cast<const signed char*>bottom_blob.data,
//                                                                            static_cast<signed char*>top_blob.data,
//                                                                            top, left, v,
//                                                                            bottom_blob.width, bottom_blob.height, w, h);
        }
        else if (elemsize == 2)
        {
            //TODO
//            cuda_copy_make_border_image_type0<unsigned short><<<grid, block>>>(static_cast<const unsigned short*>bottom_blob.data,
//                                                                               static_cast<unsigned short*>top_blob.data,
//                                                                               top, left, v,
//                                                                               bottom_blob.width, bottom_blob.height, w, h);
        }
        else if (elemsize == 4)
        {
            cuda_copy_make_border_image_type0<float><<<grid, block>>>(bottom_blob.data, top_blob.data, top, left, v,
                                                                      bottom_blob.width, bottom_blob.height, w, h);
        }
    }
    else if (type == 1)
    {
        if (elemsize == 1)
        {
            //TODO
//            cuda_copy_make_border_image_type1<signed char><<<grid, block>>>(static_cast<const signed char*>bottom_blob.data,
//                                                                            static_cast<signed char*>top_blob.data,
//                                                                            top, left,
//                                                                            bottom_blob.width, bottom_blob.height, w, h);
        }
        else if (elemsize == 2)
        {
            //TODO
//            cuda_copy_make_border_image_type1<unsigned short><<<grid, block>>>(static_cast<const unsigned short*>bottom_blob.data,
//                                                                               static_cast<unsigned short*>top_blob.data,
//                                                                               top, left,
//                                                                               bottom_blob.width, bottom_blob.height, w, h);
        }
        else if (elemsize == 4)
        {
            cuda_copy_make_border_image_type1<float><<<grid, block>>>(bottom_blob.data, top_blob.data, top, left,
                                                                      bottom_blob.width, bottom_blob.height, w, h);
        }
    }
    else if (type == 2)
    {
        if (elemsize == 1)
        {
            //TODO
//            cuda_copy_make_border_image_type2<signed char><<<grid, block>>>(static_cast<const signed char*>bottom_blob.data,
//                                                                            static_cast<signed char*>top_blob.data,
//                                                                            top, left,
//                                                                            bottom_blob.width, bottom_blob.height, w, h);
        }
        else if (elemsize == 2)
        {
            //TODO
//            cuda_copy_make_border_image_type2<unsigned short><<<grid, block>>>(static_cast<const unsigned short*>bottom_blob.data,
//                                                                               static_cast<unsigned short*>top_blob.data,
//                                                                               top, left,
//                                                                               bottom_blob.width, bottom_blob.height, w, h);

        }
        else if (elemsize == 4)
        {
            cuda_copy_make_border_image_type2<float><<<grid, block>>>(bottom_blob.data, top_blob.data, top, left,
                                                                      bottom_blob.width, bottom_blob.height, w, h);
        }
    }

    return 0;
}

template<typename T>
int copy_make_border_image_3d_cuda(const CudaMat& bottom_blob, CudaMat& top_blob, int top, int left, int type, T v,
                                   int per_channel_pad_data_size, const CudaMat& per_channel_pad_data)
{
    dim3 block;
    dim3 grid;
    int block_x = 1, block_y = 1, block_z = 1;
    int grid_x = 1, grid_y = 1, grid_z = 1;

    int w = top_blob.width;
    int h = top_blob.height;
    int channels = top_blob.channel;
    size_t elemsize = top_blob.elemsize;

    block_x = w;
    if (block_x > 32) block_x = 32;
    block_y = h;
    if (block_y > 8) block_y = 8;

    grid_x = (w - 1) / block_x + 1;
    grid_y = (h - 1) / block_y + 1;
    grid_z = (channels - 1) / block_z + 1;

    block.x = block_x;
    block.y = block_y;
    block.z = block_z;
    grid.x = grid_x;
    grid.y = grid_y;
    grid.z = grid_z;

    if (type == 0)
    {
        if (elemsize == 1)
        {
            //TODO
        }
        else if (elemsize == 2)
        {
            //TODO
        }
        else if (elemsize == 4)
        {
            cuda_copy_make_border_image_3d_type0<float><<<grid, block>>>(bottom_blob.data, top_blob.data,
                                                                         front, top, left, v,
                                                                         per_channel_pad_data.data,
                                                                         per_channel_pad_data_size,
                                                                         bottom_blob.width, bottom_blob.height,
                                                                         bottom_blob.channel,
                                                                         top_blob.width, top_blob.height,
                                                                         top_blob.channel,
                                                                         bottom_blob.pitch / sizeof(float));
        }
    }
    else if (type == 1)
    {
        if (elemsize == 1)
        {
            //TODO
        }
        else if (elemsize == 2)
        {
            //TODO
        }
        else if (elemsize == 4)
        {
            cuda_copy_make_border_image_3d_type1<float><<<grid, block>>>(bottom_blob.data, top_blob.data,
                                                                         front, left, top, v,
                                                                         per_channel_pad_data.data,
                                                                         per_channel_pad_data_size,
                                                                         bottom_blob.width, bottom_blob.height,
                                                                         bottom_blob.channel,
                                                                         top_blob.width, top_blob.height,
                                                                         top_blob.channel,
                                                                         bottom_blob.pitch / sizeof(float));
        }
    }
    else if (type == 2)
    {
        if (elemsize == 1)
        {
            //TODO
        }
        else if (elemsize == 2)
        {
            //TODO
        }
        else if (elemsize == 4)
        {
            cuda_copy_make_border_image_3d_type2<float><<<grid, block>>>(bottom_blob.data, top_blob.data,
                                                                         front, left, top, v,
                                                                         per_channel_pad_data.data,
                                                                         per_channel_pad_data_size,
                                                                         bottom_blob.width, bottom_blob.height,
                                                                         bottom_blob.channel,
                                                                         top_blob.width, top_blob.height,
                                                                         top_blob.channel,
                                                                         bottom_blob.pitch / sizeof(float));
        }
    }

    return 0;
}

} // namespace tinynn