//
// Created by lifan on 2021/2/10.
//

#ifndef DLPROJECT_MODELBIN_H
#define DLPROJECT_MODELBIN_H
#include "mat.h"
namespace tinynn
{
class DataReader;
class ModelBin
{
public:
    ModelBin();
    virtual ~ModelBin();
    //element type
    //auto 0
    //float32 1
    //float16 2
    //int8 3

    virtual Mat load(int width, int type) const = 0;
    virtual Mat load(int width, int height, int type) const;
    virtual Mat load(int width, int height, int channel, int type) const;

};
class ModelBinFromDataReaderPrivate;
class ModelBinFromDataReader: public ModelBin
{
public:
    explicit ModelBinFromDataReader(const DataReader& dr);
    virtual ~ModelBinFromDataReader();
    virtual Mat load(int width, int type) const;

private:
    ModelBinFromDataReader(const ModelBinFromDataReader&);
    ModelBinFromDataReader& operator=(const ModelBinFromDataReader&);

private:
    ModelBinFromDataReaderPrivate const* d;
};

class ModelBinFromMatArrayPrivate;
class ModelBinFromMatArray: public ModelBin
{
public:
    explicit ModelBinFromMatArray(const Mat* weights);
    virtual ~ModelBinFromMatArray();
    virtual Mat load(int width, int type) const;

private:
    ModelBinFromMatArray(const ModelBinFromMatArray&);
    ModelBinFromMatArray& operator=(const ModelBinFromMatArray&);

private:
    ModelBinFromMatArrayPrivate const* d;
};
}
#endif //DLPROJECT_MODELBIN_H
