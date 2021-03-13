//
// Created by lifan on 2021/2/15.
//

#include "c_api.h"
#include "datareader.h"
#include "modelbin.h"
#include "mat.h"
#include <cstdlib>

using tinynn::DataReader;
using tinynn::Mat;

#ifdef __cplusplus
extern "C"{
#endif

/* mat api*/
void tinynn_mat_destory(tinynn_mat_t mat)
{
    delete (Mat*)mat;
}
class ModelBinFromDataReader_c_api: public tinynn::ModelBinFromDataReader
{
public:
    ModelBinFromDataReader_c_api(tinynn_modelbin_t _mb, const DataReader& dr)
        : tinynn::ModelBinFromDataReader(dr)
    {
        mb = _mb;
    }

    virtual Mat load(int width, int type)
    {
        tinynn_mat_t m = mb->load_1d(mb ,width, type);
        Mat m2 = *(Mat*)m;
        tinynn_mat_destory(m);
        return m2;
    }

    virtual Mat load(int width, int height, int type)
    {
        tinynn_mat_t m = mb->load_2d(mb, width, height, type);
        Mat m2 = *(Mat*)m;
        tinynn_mat_destory(m);
        return m2;
    }

    virtual Mat load(int width, int height, int channel, int type)
    {
        tinynn_mat_t m = mb->load_3d(mb, width, height, channel, type);
        Mat m2 = *(Mat*)m;
        tinynn_mat_destory(m);
        return m2;
    }
public:
    tinynn_modelbin_t mb;
};
static tinynn_mat_t __tinynn_ModelBinFromDataReader_load_1d(const tinynn_modelbin_t mb, int width, int type)
{
    return (tinynn_mat_t)(new Mat(((const tinynn::ModelBinFromDataReader*)mb->pthis)->tinynn::ModelBinFromDataReader::load(width, type)));
}

static tinynn_mat_t __tinynn_ModelBinFromDataReader_load_2d(const tinynn_modelbin_t mb, int width, int height, int type)
{
    return (tinynn_mat_t)(new Mat(((const tinynn::ModelBinFromDataReader*)mb->pthis)->tinynn::ModelBin::load(width, height, type)));
}

static tinynn_mat_t __tinynn_ModelBinFromDataReader_load_3d(const tinynn_modelbin_t mb, int width, int height, int channel, int type)
{
    return (tinynn_mat_t)(new Mat(((const tinynn::ModelBinFromDataReader*)mb->pthis)->tinynn::ModelBin::load(width, height, channel, type)));
}

tinynn_modelbin_t tinynn_modelbin_create_from_datareader(const tinynn_datareader_t dr)
{
    tinynn_modelbin_t mb = (tinynn_modelbin_t)malloc(sizeof(struct __tinynn_modelbin_t));
    mb->pthis = (void*)(new ModelBinFromDataReader_c_api(mb, *(const DataReader*)dr->pthis));
    mb->load_1d = __tinynn_ModelBinFromDataReader_load_1d;
    mb->load_2d = __tinynn_ModelBinFromDataReader_load_2d;
    mb->load_3d = __tinynn_ModelBinFromDataReader_load_3d;
    return mb;
}

#ifdef __cplusplus
};/* extern "C"*/
#endif