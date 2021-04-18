//
// Created by lifan on 2021/1/25.
//

#include "paramdict.h"
#include "datareader.h"

namespace tinynn
{
#define MAX_PARAMS 32
class ParamDictPrivate
{
public:
    struct
    {
        // from ncnn
        // 0 = null
        // 1 = int/float
        // 2 = int
        // 3 = float
        // 4 = array of int/float
        // 5 = array of int
        // 6 = array of float
        int type;
        union
        {
            int i;
            float f;
        };
        Mat v;
    }params[MAX_PARAMS];
};

ParamDict::ParamDict(): d(new ParamDictPrivate)
{
    clear();
}

ParamDict::~ParamDict()
{
    delete d;
}

ParamDict::ParamDict(const ParamDict &rhs): d(new ParamDictPrivate)
{
    for (int i = 0; i < MAX_PARAMS; i++)
    {
        d->params[i].type = rhs.d->params[i].type;
        if (d->params[i].type == 1 || d->params[i].type == 2 || d->params[i].type == 3)
            d->params[i].i = rhs.d->params[i].i;
        else
            d->params[i].v = rhs.d->params[i].v;
    }
}

ParamDict &ParamDict::operator=(const ParamDict &rhs)
{
    if (this == &rhs)
        return *this;
    for (int i = 0; i < MAX_PARAMS; ++i)
    {
        d->params[i].type = rhs.d->params[i].type;
        if (d->params[i].type == 1 || d->params[i].type == 2 || d->params[i].type == 3)
            d->params[i].i = rhs.d->params[i].i;
        else
            d->params[i].v = rhs.d->params[i].v;
    }
    return *this;
}

int ParamDict::type(int id) const
{
    return d->params[id].type;
}

int ParamDict::get(int id, int def) const
{
    return d->params[id].type ? d->params[id].i: def;
}

float ParamDict::get(int id, float def) const
{
    return d->params[id].type ? d->params[id].f: def;
}

Mat ParamDict::get(int id, const Mat &def) const
{
    return d->params[id].type ? d->params[id].v: def;
}

void ParamDict::set(int id, int def)
{
    d->params[id].type = 2;
    d->params[id].i = def;
}

void ParamDict::set(int id, float def)
{
    d->params[id].type = 3;
    d->params[id].f = def;
}

void ParamDict::set(int id, Mat &def)
{
    d->params[id].type = 4;
    d->params[id].v = def;
}

void ParamDict::clear()
{
    for (int i = 0; i < MAX_PARAMS; ++i)
    {
        d->params[i].type = 0;
        d->params[i].v = Mat();
    }
}
static bool vstr_is_float(const char vstr[16])
{
    for (int i = 0; i < 16; ++i)
    {
        if (vstr[i] == '.' || tolower(vstr[i]) == 'e')
            return true;
    }
    return false;
}
static float vstr_to_float(const char vstr[16])
{
    double v = 0.0;
    const char *ptr = vstr;
    bool sign = (*ptr == '-') ? 0: 1;
    bool exponent = 0;
    if (*ptr == '+' || *ptr == '-')
        ptr++;

    //小数点前的数字
    unsigned int pow = 0;
    while (isdigit(*ptr))
    {
        pow = pow * 10 + (*ptr - '0');
        ptr++;
    }
    v = (double)pow;

    //小数点后的数字
    ptr++;
    unsigned int pow10 = 1;
    pow = 0;
    while (isdigit(*ptr))
    {
        pow = pow * 10 + (*ptr - '0');
        pow10 *= 10;
        ptr++;
    }
    v += pow / (double)pow10;
    bool fact;
    if (*ptr == 'e' || *ptr == 'E')
    {
        ptr++;
        fact = (*ptr == '-') ? 0: 1;
        if (*ptr == '+' || *ptr == '-')
            ptr++;
        pow = 0;
        while (isdigit(*ptr))
        {
            pow = pow * 10 + (*ptr - '0');
            ptr++;
        }
        double scale = 1.0;
        while (pow > 8)
        {
            pow -= 8;
            scale *= 1e8;
        }
        while (pow > 0)
        {
            pow--;
            scale *= 1e1;
        }
        v = fact ? v * scale: v / scale;
    }
    return sign ? (float)v: (float)-v;
}
//读取失败返回-1
int ParamDict::load_param(const DataReader &dr)
{
    clear();

    int id = 0;

    while (dr.scan("%d=", &id) == 1)
    {
        bool is_array = (id <= -23300) ? 1: 0;
        if (is_array)
        {
            int len = 0;
            bool is_float;
            id = -id - 23300;
            if (dr.scan("%d", &len) == 1)
            {
                d->params[id].v.create(len);
//                float *ptr = (float*)d->params[id].v.data;
                for (int i = 0; i < len; ++i) {
                    char vstr[16];
                    if (dr.scan(",%15[^,\n ]", vstr) == 1)
                    {
                        if (vstr_is_float(vstr) == 1)
                        {
                            float* ptr = d->params[id].v;
                            ptr[i] = vstr_to_float(vstr);
                            is_float = 1;
                        }
                        else
                        {
                            int* ptr = d->params[id].v;
                            sscanf(vstr, "%d", &ptr[i]);
                            is_float = 0;
                        }
                    }
                    else
                    {
                        std::cout << "ERROR: Paramdict array read" << std::endl;
                        return -1;
                    }
                }
            }
            else
            {
                std::cout << "ERROR: Paramdict array read" << std::endl;
                return -1;
            }
            //set type
            d->params[id].type = is_float ? 6: 5;
        }
        else
        {
            char vstr[16];
            bool is_float;
            if (dr.scan("%s", vstr) == 1)
            {
                if (vstr_is_float(vstr) == 1)
                {
                    d->params[id].f = vstr_to_float(vstr);
                    is_float = true;
                }
                else
                {
                    sscanf(vstr, "%d", &d->params[id].i);
                    is_float = false;
                }
            }
            else
            {
                std::cout << "ERROR: Paramdict num read" << std::endl;
                return -1;
            }
            //set type
            d->params[id].type = is_float ? 3: 2;
        }
    }
    return 0;
}

void ParamDict::temptest() {
//    const char vstr[16] = "-123";
//    bool res = vstr_is_float(vstr);
//    float res = vstr_to_float(vstr);
//    printf("%f\n", res);

    const unsigned char* mem = (const unsigned char*)"-23303=5,0.1,0.2,0.4,0.8,1.0 0=100 1=1.25000";
    DataReaderFromMemory d(mem);
    load_param(d);
    for (int i = 0; i < 5; ++i) {
        printf("%f\n", this->d->params[3].v[i]);
    }
    printf("%d\n", this->d->params[0].i);
    printf("%f\n", this->d->params[1].f);
}

}