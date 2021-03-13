//
// Created by lifan on 2021/2/16.
//

#include "cpu.h"
#include <cstring>
#include <cstdio>
namespace tinynn
{
static int get_cpucount()
{
    int count = 0;
    FILE* fp = fopen("/proc/cpuinfo", "rb");

    if (!fp)
        return 1;

    char line[1024];
    while (!feof(fp))
    {
        char* s = fgets(line, 1024, fp);
        if (!s)
            break;

        if (memcmp(line, "processor", 9) == 0)
            count++;
    }
    fclose(fp);
    if (count < 1)
        return 1;

    return count;
}
static int g_cpucount = get_cpucount();
int get_cpu_count()
{
    return g_cpucount;
}
}