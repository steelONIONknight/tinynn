//
// Created by lifan on 2021/5/17.
//

#ifndef TINYNN_CONVOLUTION_CUDA_H
#define TINYNN_CONVOLUTION_CUDA_H
#include "../convolution.h"

namespace tinynn
{
class Convolution_cuda : virtual public Convolution
{
public:
    Convolution_cuda();
    ~Convolution_cuda();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int create_pipeline(const Option& opt);
    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

protected:
    void make_padding(const CudaMat& bottom_blob, CudaMat& bottom_blob_bordered, const Option& opt) const;

public:
    int num_output;

    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;

    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    float pad_value;

    int bias_term;
    int weight_data_size;

    //0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    CudaMat activation_param;

    //model
    CudaMat weight_data;
    CudaMat bias_data;

private:
    CudaAllocator* _cudaAllocator;

};
} // namespace tinynn
#endif //TINYNN_CONVOLUTION_CUDA_H
