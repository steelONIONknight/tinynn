//
// Created by lifan on 2021/3/25.
//

#include "innerproduct.h"

namespace tinynn
{

InnerProduct::InnerProduct()
{
    one_blob_only = true;
    support_inplace = false;
}

int InnerProduct::load_param(const ParamDict &pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());
}

int InnerProduct::load_model(const ModelBin &mb)
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

int InnerProduct::create_pipeline(const Option &op)
{
    //now do nothing
    return 0;
}

int InnerProduct::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
{
    int width = bottom_blob.width;
    int height = bottom_blob.height;
    int channels = bottom_blob.channel;
    size_t elemsize = bottom_blob.elemsize;
    int size = width * height;

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        for (int q = 0; q < channels; q++)
        {
            const float* w = (const float*)weight_data + p * size * channels + q * size;
            const float* m = bottom_blob.refer_channel(q);

            for (int i = 0; i < size; ++i)
            {
                sum += w[i] * m[i];
            }
        }

        if (activation_type == 1)
        {
            sum = std::max(sum, 0.f);
        }
        else if (activation_type == 2)
        {
            float slope = activation_params[0];
            sum = sum > 0.f ? sum : sum * slope;
        }
        else if (activation_type == 3)
        {
            float min = activation_params[0];
            float max = activation_params[1];

            if (sum < min)
                sum = min;
            if (sum > max)
                sum = max;
        }
        else if (activation_type == 4)
        {
            sum = static_cast<float>(1.f / 1.f + exp(-sum));
        }
        else if (activation_type == 5)
        {
            sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
        }

        top_blob[p] = sum;
    }

    return 0;
}


}