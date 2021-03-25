//
// Created by lifan on 2021/2/26.
//

#ifndef DLPROJECT_CAST_H
#define DLPROJECT_CAST_H
#include "../layer.h"

namespace tinynn
{
class Cast: public Layer
{
public:
    Cast();

    virtual int load_param(const ParamDict& pd);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    //element type
    //0 auto
    //1 float32
    //2 float16
    int type_from;
    int type_to;
};
}
#endif //DLPROJECT_CAST_H
