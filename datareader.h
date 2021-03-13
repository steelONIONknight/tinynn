//
// Created by lifan on 2021/1/18.
//

#ifndef DLPROJECT_DATAREADER_H
#define DLPROJECT_DATAREADER_H
#include <cstdio>
#include <cstring>
namespace tinynn
{
class DataReader
{
public:
    DataReader();
    virtual ~DataReader();
    virtual int scan(const char* format, void* p) const;
    virtual size_t read(void* buffer, size_t size) const;
};
class DataReaderFromMemoryPrivate;
class DataReaderFromMemory: public DataReader
{
public:
    explicit DataReaderFromMemory(const unsigned char* &mem);
    virtual ~DataReaderFromMemory();
    virtual int scan(const char* format, void* p) const;
    virtual size_t read(void* buffer, size_t size) const;

private:
    //什么用处
    //TODO
    DataReaderFromMemory(const DataReaderFromMemory&);
    DataReaderFromMemory& operator=(const DataReaderFromMemory&);

private:
    DataReaderFromMemoryPrivate* const d;
};
}
#endif //DLPROJECT_DATAREADER_H
