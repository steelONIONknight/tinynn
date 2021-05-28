//
// Created by lifan on 2021/5/17.
//

#include "padding_cuda.h"

namespace tinynn
{
template<typename T>
int copy_make_border_image_cuda(const CudaMat& bottom_blob, CudaMat& top_blob, int top, int left, int type, T v);

template<typename T>
int copy_make_border_image_3d_cuda(const CudaMat& bottom_blob, CudaMat& top_blob, int top, int left, int type, T v,
                                   int per_channel_pad_data_size, const CudaMat& per_channel_pad_data);

Padding_cuda::Padding_cuda()
{
    one_blob_only = true;
    support_inplace = false;
    support_cuda = true;

    //default use GPU 0
    CudaDevice cudev(0);
    _cudaAllocator = new CudaAllocator(&cudev);

    per_channel_pad_data.cudaAllocator = _cudaAllocator;
}

Padding_cuda::~Padding_cuda()
{
    per_channel_pad_data.release();
    delete _cudaAllocator;
}

int Padding_cuda::load_param(const ParamDict &pd)
{
    top = pd.get(0, 0);
    bottom = pd.get(1, 0);
    left = pd.get(2, 0);
    right = pd.get(3, 0);
    type = pd.get(4, 0);
    value = pd.get(5, 0.f);
    per_channel_pad_data_size = pd.get(6, 0);
    front = pd.get(7, 0);
    behind = pd.get(8, 0);

    if (top == -233 && bottom == -233 && left == -233 && right == -233)
    {
        one_blob_only = false;
    }
    if (top == -234 && bottom == -234 && left == -234 && right == -234)
    {
        one_blob_only = false;
    }

    return 0;
}

int Padding_cuda::load_model(const ModelBin &mb)
{
    if (per_channel_pad_data_size)
    {
        per_channel_pad_data = mb.load(per_channel_pad_data_size, 1);
    }

    return 0;
}

int Padding_cuda::forward(const CudaMat &bottom_blob, CudaMat &top_blob, const Option &opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.width;
    int h = bottom_blob.height;
    int channels = bottom_blob.channel;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w + left + right;
    int outh = h + top + bottom;
    int outc = channels + front + behind;

    if (dims == 1 || dims == 2)
    {
        if (dims == 1)
        {
            top_blob.create(outw, elemsize, opt.blob_cuda_allocator);
            if (top_blob.empty())
                return -100;
        }
        else
        {
            top_blob.create(outw, outh, elemsize, opt.blob_cuda_allocator);
            if (top_blob.empty())
                return -100;
        }

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
            copy_make_border_image_cuda<float>(bottom_blob, top_blob, top, left, type, value);
        }
    }
    else if (dims == 3)
    {
        top_blob.create(outw, outh, outc, elemsize, opt.blob_cuda_allocator);
        if (top_blob.empty())
            return -100;

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
            copy_make_border_image_3d_cuda<float>(bottom_blob, top_blob, top, left, type, value, per_channel_pad_data_size, per_channel_pad_data);
        }
    }
    return 0;
}

int Padding_cuda::forward(const std::vector<CudaMat> &bottom_blobs, std::vector<CudaMat> &top_blobs, const Option &opt) const
{
    throw std::runtime_error("Not implement");
    return 0;
}

} // namespace tinynn