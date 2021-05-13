//
// Created by lifan on 2021/5/4.
//

#include "convolution.h"
#include "../layer_type.h"

namespace tinynn
{

Convolution::Convolution()
{
    one_blob_only = true;
    support_inplace = false;
}

int Convolution::load_param(const ParamDict &pd)
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

int Convolution::load_model(const ModelBin &mb)
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

int Convolution::create_pipeline(const Option &opt)
{

    return 0;
}
//TODO
int Convolution::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
{
    //conv with N*N kernel
    //value = value + bias


    //flattered blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.width * bottom_blob.elempack == num_input)
        {
            tinynn::Layer* op = tinynn::create_layer(tinynn::LayerType::InnerProduct);

            tinynn::ParamDict pd;
            pd.get(0, num_output);
            pd.get(1, bias_term);
            pd.get(2, weight_data_size);
            pd.get(9, activation_type);
            pd.get(10, activation_params);

            op->load_param(pd);

            tinynn::Mat weights[4];
            weights[0] = weight_data;
            weights[1] = bias_data;


            op->load_model(ModelBinFromMatArray(weights));

            op->create_pipeline(opt);

            op->forward(bottom_blob, top_blob, opt);

            op->destroy_pipeline(opt);

            delete op;

            return 0;
        }
    }


}

void Convolution::make_padding(const Mat &bottom_blob, Mat &bottom_blob_bordered, const Option &opt) const
{
    int w = bottom_blob.width;
    int h = bottom_blob.height;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        //tensorflow padding = SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        //onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

}