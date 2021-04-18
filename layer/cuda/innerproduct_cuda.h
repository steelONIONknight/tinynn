//
// Created by lifan on 2021/3/26.
//

#ifndef TINYNN_INNERPRODUCT_CUDA_H
#define TINYNN_INNERPRODUCT_CUDA_H
#include "../innerproduct.h"

namespace tinynn
{
class InnerProduct_cuda: virtual public InnerProduct
{
public:
    InnerProduct_cuda();
    ~InnerProduct_cuda();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int create_pipeline(const Option& opt);
    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

public:
    int num_output;
    int bias_term;
    int weight_data_size;

    int activation_type;
    CudaMat activation_params;

    //model
    CudaMat weight_data;
    CudaMat bias_data;

private:
    CudaAllocator* _cudaAllocator;

};
} // namespace tinynn
#endif //TINYNN_INNERPRODUCT_CUDA_H
