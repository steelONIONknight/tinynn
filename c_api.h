//
// Created by lifan on 2021/2/15.
//

#ifndef DLPROJECT_C_API_H
#define DLPROJECT_C_API_H

#include <cstddef>

#ifdef __cplusplus
extern "C"{
#endif

/* mat api*/
typedef struct __tinynn_mat_t* tinynn_mat_t;
void tinynn_mat_destory(tinynn_mat_t mat);

/* datareader api*/
typedef struct __tinynn_datereader_t* tinynn_datareader_t;
struct __tinynn_datereader_t
{
    void* pthis;
    int (*scan)(tinynn_datareader_t dr, const char* format, void* p);
    size_t (*read)(tinynn_datareader_t dr, void* buffer, size_t size);
};

/* modelbin api*/
typedef struct __tinynn_modelbin_t* tinynn_modelbin_t;
struct __tinynn_modelbin_t
{
    void* pthis;

    tinynn_mat_t (*load_1d)(const tinynn_modelbin_t mb, int width, int type);
    tinynn_mat_t (*load_2d)(const tinynn_modelbin_t mb, int width, int height, int type);
    tinynn_mat_t (*load_3d)(const tinynn_modelbin_t mb, int width, int height, int channel, int type);
};

tinynn_modelbin_t tinynn_modelbin_create_from_datareader(const tinynn_datareader_t dr);

#ifdef __cplusplus
};/* extern "C" */
#endif
#endif //DLPROJECT_C_API_H
