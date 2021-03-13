//
// Created by lifan on 2021/2/25.
//

#ifndef DLPROJECT_ABSVAL_H
#define DLPROJECT_ABSVAL_H
#include "../layer.h"

namespace tinynn
{
class AbsVal: public Layer
{
public:
    AbsVal();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

};
}
#endif //DLPROJECT_ABSVAL_H
