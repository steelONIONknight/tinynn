//
// Created by lifan on 2021/3/25.
//

#ifndef TINYNN_INNERPRODUCT_H
#define TINYNN_INNERPRODUCT_H
#include "../layer.h"

namespace tinynn
{
class InnerProduct: public Layer
{
public:
    InnerProduct();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int create_pipeline(const Option& op);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int num_output;
    int bias_term;
    int weight_data_size;

    //0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    //model
    Mat weight_data;
    Mat bias_data;
};
}
#endif //TINYNN_INNERPRODUCT_H
