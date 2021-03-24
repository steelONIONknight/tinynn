//
// Created by lifan on 2021/3/14.
//

#ifndef TINYNN_LAYER_DECLARATION_H
#define TINYNN_LAYER_DECLARATION_H
#include "layer/absval.h"
#include "layer/cuda/absval_cuda.h"

namespace tinynn
{
class AbsVal_final: virtual public AbsVal, virtual public AbsVal_cuda
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        { int ret = AbsVal::create_pipeline(opt); if (ret) return ret; }
        { int ret = AbsVal_cuda::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        { int ret = AbsVal::destroy_pipeline(opt); if (ret) return ret; }
        { int ret = AbsVal_cuda::destroy_pipeline(opt); if (ret) return ret;}
        return 0;
    }
};
DEFINE_LAYER_CREATOR(AbsVal_final)
}//namespace tinynn
#endif //TINYNN_LAYER_DECLARATION_H
