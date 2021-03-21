//
// Created by lifan on 2021/2/16.
//

#include "option.h"
#include "cpu.h"

namespace tinynn
{

Option::Option()
{
    lightmode = true;
    num_threads = get_cpu_count();

    blob_allocator = nullptr;
    workspace_allocator = nullptr;

    blob_cuda_allocator = nullptr;
    workspace_cuda_allocator = nullptr;

    openmp_blocktime = 20;

    use_winograd_convolution = true;
    use_sgemm_convolution = true;

    use_fp16_packed = true;
    use_fp16_storage = true;
    use_fp16_arithmetic = true;

    use_packing_layout = true;
    use_shader_pack8 = false;

    use_subgroup_basic = false;
    use_subgroup_vote = false;
    use_subgroup_ballot = false;
    use_subgroup_shuffle = false;
}
}