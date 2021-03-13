//
// Created by lifan on 2021/2/16.
//

#ifndef DLPROJECT_OPTION_H
#define DLPROJECT_OPTION_H
namespace tinynn
{
class Allocator;
class Option
{
public:
    Option();

public:
    // light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    bool lightmode;

    // thread count
    // default value is the one returned by get_cpu_count()
    int num_thread;

    //blob memory allocator
    Allocator* blob_allocator;

    //workspace memory allocator
    Allocator* workspace_allocator;


    // the time openmp threads busy-wait for more work before going to sleep
    // default value is 20ms to keep the cores enabled
    // without too much extra power consumption afterwards
    int openmp_blocktime;

    // enable winograd convolution optimization
    // improve convolution 3x3 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_winograd_convolution;

    // enable sgemm convolution optimization
    // improve convolution 1x1 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_sgemm_convolution;

    //enable option for gpu inference
    bool use_fp16_packed;
    bool use_fp16_storage;
    bool use_fp16_arithmetic;

    // enable simd-friendly packed memory layout
    // improve all operator performance on all arm devices, will consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_packing_layout;

    bool use_shader_pack8;

    //subgroup option
    bool use_subgroup_basic;
    bool use_subgroup_vote;
    bool use_subgroup_ballot;
    bool use_subgroup_shuffle;


    bool use_reserved_0;
    bool use_reserved_1;
    bool use_reserved_2;
    bool use_reserved_3;
    bool use_reserved_4;
    bool use_reserved_5;
    bool use_reserved_6;
    bool use_reserved_7;
    bool use_reserved_8;
    bool use_reserved_9;
    bool use_reserved_10;
    bool use_reserved_11;
};
}
#endif //DLPROJECT_OPTION_H
