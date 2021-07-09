//
// Created by lifan on 2021/5/17.
//

#include "convolution_cuda.h"

namespace tinynn
{

Convolution_cuda::Convolution_cuda()
{
    one_blob_only = true;
    support_inplace = false;
    support_cuda = true;
    //default use GPU 0
    CudaDevice cudev(0);
    _cudaAllocator = new CudaAllocator(&cudev);
    //给GPU变量指定CudaAllocator
    activation_param.cudaAllocator = _cudaAllocator;
    weight_data.cudaAllocator = _cudaAllocator;
    bias_data.cudaAllocator = _cudaAllocator;
}

Convolution_cuda::~Convolution_cuda()
{
    activation_param.release();
    weight_data.release();
    bias_data.release();
    delete _cudaAllocator;
}

int Convolution_cuda::load_param(const ParamDict &pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    return 0;
}
//TODO
//有点棘手？
//bottom_blob需要padding，在host端完成？
int Convolution_cuda::load_model(const ModelBin &mb)
{
//    weight_data = mb.load(weight_data_size, 0);
//    if (weight_data.empty())
//        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Convolution_cuda::create_pipeline(const Option &opt)
{

    return 0;
}

int Convolution_cuda::forward(const CudaMat &bottom_blob, CudaMat &top_blob, const Option &opt) const
{

}

void Convolution_cuda::make_padding(const CudaMat &bottom_blob, CudaMat &bottom_blob_bordered, const Option &opt) const
{

}


}