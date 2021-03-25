//
// Created by lifan on 2021/2/18.
//

#ifndef DLPROJECT_LAYER_H
#define DLPROJECT_LAYER_H
#include "mat.h"
#include "paramdict.h"
#include "modelbin.h"
#include "datareader.h"
#include "option.h"
#include <cmath>
#include <vector>
#include <string>

namespace tinynn
{
class Layer
{
public:
    Layer();
    virtual ~Layer();

    //加载parsed dict中的layer的参数，具体来讲就是算子的参数
    //成功返回0
    virtual int load_param(const ParamDict& pd);

    //加载模型，即加载模型的权重
    //成功返回0
    virtual int load_model(const ModelBin& mb);

    //layer implementation specific setup
    //return 0 if success
    virtual int create_pipeline(const Option& opt);

    //layer implementation specific clean
    //return 0 if success
    virtual int destroy_pipeline(const Option& opt);

public:
    //one input blob and one output blob
    bool one_blob_only;

    //support inplace inference
    bool support_inplace;

    //accept input blob with packed storage
    bool support_packing;

    bool support_fp16_storage;

    //shader image storage
    bool support_image_storage;

    //shader tensor storage
    bool support_tensor_storage;

    bool support_cuda{false};


public:
    //implement inference
    //return 0 if success
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    //implement inplace inference
    //return 0 if success
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    const CudaDevice* cudev;
    virtual int load_model(const CudaModelBinFromMatArray& /*mb*/);

    virtual int forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const;
    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

    virtual int forward_inplace(std::vector<CudaMat>& bottom_top_blobs, const Option& opt) const;
    virtual int forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const;

public:
    //自定义用户数据
    void* userdata;

    //layer type index
    int type_index;

    //layer type name
    std::string type;

    //layer name
    std::string name;

    //blob index which this layer needs as input
    std::vector<int> bottoms;

    //blob index which this layer produces as output
    std::vector<int> tops;

    //shape hint
    std::vector<Mat> bottom_shapes;

    std::vector<Mat> top_shapes;
};

//layer的工厂方法
//layer factory function
typedef Layer* (*layer_creator_func)(void*);
typedef void (*layer_destroy_func)(Layer*, void*);

struct layer_registry_entry
{
    const char* name;
    layer_creator_func creator;
};

struct custom_layer_registry_entry
{
    const char* name;
    layer_creator_func creator;
    layer_destroy_func destroyer;

    void* userdata;
};
//get layer type from type name
int layer_to_index(const char* type);

//create layer from type name
Layer* create_layer(const char* type);

//create layer from layer type
Layer* create_layer(int index);


#define DEFINE_LAYER_CREATOR(name)                           \
    ::tinynn::Layer* name##_layer_creator(void* /*userdata*/)\
    {                                                        \
        return new name;                                     \
    }

#define DEFINE_LAYER_DESTROYER(name)                                       \
    void name##_layer_destroyer(::tinynn::Layer* layer, void* /*userdata*/)\
    {                                                                      \
        delete layer;                                                      \
    }

}//namespace tinynn

#endif //DLPROJECT_LAYER_H
