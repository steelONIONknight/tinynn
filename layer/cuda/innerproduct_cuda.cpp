//
// Created by lifan on 2021/3/26.
//

#include "innerproduct_cuda.h"

namespace tinynn
{
int innerproduct_forward(const CudaMat& bottom_blob, const CudaMat& weight, const CudaMat& activation_params,
                         const CudaMat& bias_data, CudaMat& top_blob, int num_output, int bias_term, int weight_data_size,
                         int activation_type);

InnerProduct_cuda::InnerProduct_cuda()
{
    one_blob_only = true;
    support_inplace = false;
    support_cuda = true;
    //默认使用0号GPU
    CudaDevice cudev(0);
    _cudaAllocator = new CudaAllocator(&cudev);

    //给GPU变量指定CudaAllocator
    activation_params.cudaAllocator = _cudaAllocator;
    weight_data.cudaAllocator = _cudaAllocator;
    bias_data.cudaAllocator = _cudaAllocator;
}

InnerProduct_cuda::~InnerProduct_cuda()
{
    activation_params.release();
    weight_data.release();
    bias_data.release();
    delete _cudaAllocator;
}

int InnerProduct_cuda::load_param(const ParamDict &pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    return 0;
}

int InnerProduct_cuda::load_model(const ModelBin &mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int InnerProduct_cuda::create_pipeline(const Option &opt)
{
    return 0;
}

int InnerProduct_cuda::forward(const CudaMat &bottom_blob, CudaMat &top_blob, const Option &opt) const
{
    top_blob.create(num_output, bottom_blob.elemsize, opt.blob_cuda_allocator);

    if (top_blob.empty())
        return -100;

    return innerproduct_forward(bottom_blob, weight_data, activation_params, bias_data, top_blob, num_output,
                                bias_term, weight_data_size, activation_type);

}


} //namespace tinynn