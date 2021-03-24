//
// Created by lifan on 2021/2/18.
//

#include "layer.h"
#include "layer_declaration.h"

namespace tinynn
{
Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
    support_packing = false;
    support_fp16_storage = false;
    support_image_storage = false;
    support_tensor_storage = false;
    type_index = -1;
    userdata = nullptr;
}
Layer::~Layer()
{
}

int Layer::load_parm(const ParamDict& /*pd*/)
{
    return 0;
}

int Layer::load_model(const ModelBin& /*mb*/)
{
    return 0;
}

int Layer::create_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Layer::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Layer::forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const
{
    if (!support_inplace)
        return -1;

    top_blobs = bottom_blobs;
    for (int i = 0; i < (int)bottom_blobs.size(); ++i)
    {
        top_blobs[i] = bottom_blobs[i].clone(opt.blob_allocator);
        if (top_blobs[i].empty())
            return -100;
    }

    return forward_inplace(top_blobs, opt);
}

int Layer::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
{
    if (!support_inplace)
        return -1;

    top_blob = bottom_blob.clone(opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    return forward_inplace(top_blob, opt);
}

int Layer::forward_inplace(std::vector<Mat>& /*bottom_top_blob*/, const Option &/*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(Mat& /*bottom_top_blob*/, const Option &/*opt*/) const
{
    return -1;
}

int Layer::load_model(const CudaModelBinFromMatArray &)
{
    return 0;
}

int Layer::forward(const std::vector<CudaMat> &/*bottom_blobs*/, std::vector<CudaMat> &/*top_blobs*/, const Option &/*opt*/) const
{
    return -1;
}

int Layer::forward(const CudaMat &/*bottom_blob*/, CudaMat &/*top_blob*/, const Option &/*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(std::vector<CudaMat> &/*bottom_top_blob*/, const Option &/*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(CudaMat &/*bottom_top_blob*/, const Option &/*opt*/) const
{
    return -1;
}

//TODO
static const layer_registry_entry layer_registry[] = {
#include "layer_registry.h"
};

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);



int layer_to_index(const char* type)
{
    for (int i = 0; i < layer_registry_entry_count; ++i)
    {
        if (strcmp(type, layer_registry[i].name) == 0)
            return i;
    }
    return -1;
}

Layer* create_layer(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return nullptr;

    return create_layer(index);
}

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return nullptr;

    layer_creator_func layer_creator = nullptr;

    layer_creator = layer_registry[index].creator;

    if (!layer_creator)
        return nullptr;

    Layer* layer = layer_creator(nullptr);
    layer->type_index = index;
    return layer;
}
}

