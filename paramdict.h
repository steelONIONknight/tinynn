//
// Created by lifan on 2021/1/20.
//

#ifndef DLPROJECT_PARAMDICT_H
#define DLPROJECT_PARAMDICT_H
#include "mat.h"


namespace tinynn
{
class ParamDictPrivate;
class DataReader;
class ParamDict
{
public:
    ParamDict();
    virtual ~ParamDict();
    ParamDict(const ParamDict&);
    ParamDict& operator=(const ParamDict&);

    int type(int id) const;

    int get(int id, int def) const;
    float get(int id, float def) const;
    Mat get(int id, const Mat& def) const;

    void set(int id, int def);
    void set(int id, float def);
    void set(int id, Mat& def);



//just test
//don't use this function
    void temptest();



protected:
    //TODO
    //friend class Net;
    void clear();
    int load_param(const DataReader& dr);
//    TODO
//    int load_param_bin(const DataReader& dr);
private:
    ParamDictPrivate *const d;
};
}
#endif //DLPROJECT_PARAMDICT_H
