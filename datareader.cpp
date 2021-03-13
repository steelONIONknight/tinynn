//
// Created by lifan on 2021/1/18.
//
#include "datareader.h"

namespace tinynn
{

DataReader::DataReader()
{
}

DataReader::~DataReader()
{
}

int DataReader::scan(const char* /*format*/, void* /*p*/) const
{
    return 0;
}

size_t DataReader::read(void* /*buffer*/, size_t /*size*/) const
{
    return 0;
}

class DataReaderFromMemoryPrivate
{
public:
    DataReaderFromMemoryPrivate(const unsigned char* &_mem): mem(_mem){}
    const unsigned char* &mem;
};

DataReaderFromMemory::DataReaderFromMemory(const unsigned char *&mem)
    :d(new DataReaderFromMemoryPrivate(mem))
{
}
DataReaderFromMemory::DataReaderFromMemory(const DataReaderFromMemory &): d(0) {}
DataReaderFromMemory& DataReaderFromMemory::operator=(const DataReaderFromMemory&)
{
    return *this;
}
DataReaderFromMemory::~DataReaderFromMemory()
{
    delete[] d;
}
//例子
//*format = "%d="
//%s%%n -> "%d=%n"
//*p = 0
int DataReaderFromMemory::scan(const char *format, void *p) const {
    int format_len = strlen(format);
    char *format_with_n = new char[format_len + 4];
    int nconsumed = 0;
    int nscan = 0;
    sprintf(format_with_n, "%s%%n", format);
    nscan = sscanf((const char *)d->mem, format_with_n, p, &nconsumed);
    //在mem中扫描了nconsumed个字符,等号左边的值写入p
    d->mem += nconsumed;
    delete[] format_with_n;

    return nconsumed > 0 ? nscan : 0;
}
//将mem的数据拷贝到buffer
size_t DataReaderFromMemory::read(void *buffer, size_t size) const {
    memcpy(buffer, d->mem, size);
    d->mem += size;
    return size;
}

}