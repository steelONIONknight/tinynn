//
// Created by lifan on 2021/5/17.
//

#ifndef TINYNN_PADDING_CUDA_H
#define TINYNN_PADDING_CUDA_H
#include "../padding.h"

namespace tinynn
{
class Padding_cuda : virtual public Padding
{
public:
    Padding_cuda();
    ~Padding_cuda();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const;

public:
    // -233 = dynamic offset from reference blob
    int top;
    int bottom;
    int left;
    int right;
    int type; // 0=CONSTANT 1=REPLICATE 2=REFLECT
    float value;
    int front;
    int behind;

    //per channel pad value
    int per_channel_pad_data_size;
    CudaMat per_channel_pad_data;

private:
    CudaAllocator* _cudaAllocator;
};
}
#endif //TINYNN_PADDING_CUDA_H
