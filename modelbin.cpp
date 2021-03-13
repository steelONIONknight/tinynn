//
// Created by lifan on 2021/2/10.
//

#include "modelbin.h"
#include "datareader.h"
#include <cstring>
#include <cstdio>
#include <vector>

#define LOGE(...) do {\
    fprintf(stderr, ##__VA_ARGS__); fprintf(stderr, "\n");} while(0)
namespace tinynn
{
class ModelBinFromDataReaderPrivate
{
public:
    ModelBinFromDataReaderPrivate(const DataReader& _dr): dr(_dr){}
    const DataReader& dr;
};

ModelBin::ModelBin()
{
}

ModelBin::~ModelBin()
{
}

Mat ModelBin::load(int width, int height, int type) const
{
    Mat m = load(width * height, type);
    if (m.empty())
        return m;
    return m.reshape(width, height);
}

Mat ModelBin::load(int width, int height, int channel, int type) const
{
    Mat m = load(width * height * channel, type);
    if (m.empty())
        return m;
    return m.reshape(width, height, channel);
}

ModelBinFromDataReader::ModelBinFromDataReader(const DataReader &_dr)
    : ModelBin(), d(new ModelBinFromDataReaderPrivate(_dr))
{
}

ModelBinFromDataReader::ModelBinFromDataReader(const ModelBinFromDataReader&):d(0)
{
}

ModelBinFromDataReader::~ModelBinFromDataReader()
{
    delete d;
}

ModelBinFromDataReader& ModelBinFromDataReader::operator=(const ModelBinFromDataReader &)
{
    return *this;
}

Mat ModelBinFromDataReader::load(int width, int type) const
{
    if (type == 0)
    {
        size_t nread;
        union
        {
            struct
            {
                unsigned char f0;
                unsigned char f1;
                unsigned char f2;
                unsigned char f3;
            };
            unsigned int tag;
        }flag_struct;
        //flag_struct的数值代表的目前意义不明
        unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;
        nread = d->dr.read(&flag_struct, sizeof(flag_struct));
        if (nread != sizeof(flag_struct))
        {
            LOGE("ModelBin read flag_struct failed %zd", nread);
            return Mat();
        }
        //tag指代的数字还不明确作用,之后需要搞明白
        //TODO
        if (flag_struct.tag == 0x01306B47)
        {
            //读取float16半精度的模型权重
            size_t align_sz = align_size(width * sizeof(unsigned short), 4);
            std::vector<unsigned short> float16_weights;
            float16_weights.resize(align_sz);
            nread = d->dr.read(float16_weights.data(), align_sz);
            if (nread != align_sz)
            {
                LOGE("ModelBin read float16_weights failed %zd", nread);
                return Mat();
            }
            Mat m;
            m = Mat::from_float16(float16_weights.data(), width);
            return m;
        }
        else if (flag_struct.tag == 0x000D4B38)
        {
            //读取int8的模型权重
            //TODO
        }
        else if (flag_struct.tag == 0x0002C056)
        {
            //读取float32单精度的模型权重
            Mat m(width);
            if (m.empty())
                return Mat();
            nread = d->dr.read(m, width * sizeof(float));
            if (nread != width * sizeof(float))
            {
                LOGE("ModelBin read float32_weights failed %zd", nread);
                return Mat();
            }
            return m;
        }
        Mat m(width);
        if (m.empty())
            return Mat();

        if (flag != 0)
        {
            float quantization_value[256];
            nread = d->dr.read(quantization_value, 256 * sizeof(float));
            if (nread != 256 * sizeof(float))
            {
                LOGE("ModelBin read quantization_value failed %zd", nread);
                return Mat();
            }
            size_t align_weight_data_sz = align_size(width * sizeof(unsigned char), 4);
            std::vector<unsigned char> index_arr;
            index_arr.resize(align_weight_data_sz);
            nread = d->dr.read(index_arr.data(), align_weight_data_sz);
            if (nread != align_weight_data_sz)
            {
                LOGE("ModelBin read index_arr failed %zd", nread);
                return Mat();
            }
            float* ptr = m;
            for (int i = 0; i < width; ++i)
            {
                ptr[i] = quantization_value[index_arr[i]];
            }
        }
        else if (flag_struct.f0 == 0)
        {
            //row data
            nread = d->dr.read(m, width * sizeof(float));
            if (nread != width * sizeof(float))
            {
                LOGE("ModelBin read weight_data failed %zd", nread);
                return Mat();
            }
        }

        return m;
    }
    else if (type == 1)
    {
        Mat m(width);
        if (m.empty())
            return Mat();

        size_t nread;
        nread = d->dr.read(m, width * sizeof(float));
        if (nread != width * sizeof(float))
        {
            LOGE("ModelBin read float32_weights failed %zd", nread);
            return Mat();
        }
        return m;
    }
    else
    {
        LOGE("ModelBin load type %d not implemented", type);
        return Mat();
    }
}
class ModelBinFromMatArrayPrivate
{
public:
    ModelBinFromMatArrayPrivate(const Mat* _weights): weights(_weights)
    {}
    mutable const Mat* weights;
};

ModelBinFromMatArray::ModelBinFromMatArray(const Mat *weights):
        ModelBin(), d(new ModelBinFromMatArrayPrivate(weights))
{
}

ModelBinFromMatArray::~ModelBinFromMatArray()
{
    delete d;
}

Mat ModelBinFromMatArray::load(int /*width*/, int /*type*/) const
{
    if (!d->weights)
        return Mat();

    Mat m = d->weights[0];
    d->weights++;
    return m;
}

ModelBinFromMatArray::ModelBinFromMatArray(const ModelBinFromMatArray &): d(nullptr)
{
}

ModelBinFromMatArray & ModelBinFromMatArray::operator=(const ModelBinFromMatArray &)
{
    return *this;
}

}