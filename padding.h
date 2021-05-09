//
// Created by lifan on 2021/5/4.
//

#ifndef TINYNN_PADDING_H
#define TINYNN_PADDING_H
#include "layer.h"

namespace tinynn
{

class Padding: public Layer
{
public:
    Padding();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

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
    Mat per_channel_pad_data;

};

}  // namespace tinynn
#endif //TINYNN_PADDING_H
