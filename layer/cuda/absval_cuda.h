//
// Created by lifan on 2021/2/28.
//

#ifndef DLPROJECT_ABSVAL_CUDA_H
#define DLPROJECT_ABSVAL_CUDA_H
#include "../absval.h"

namespace tinynn
{
class AbsVal_cuda: public AbsVal
{
public:
    AbsVal_cuda();

    virtual int forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const;
};
}
#endif //DLPROJECT_ABSVAL_CUDA_H
