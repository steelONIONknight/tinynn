//
// Created by lifan on 2021/5/4.
//

#ifndef TINYNN_LAYER_TYPE_H
#define TINYNN_LAYER_TYPE_H

namespace tinynn
{
namespace LayerType
{
enum LayerType
{
#include "layer_type_enum.h"
    CustomBit = (1 << 8),
};

} // namespace LayerType
} // namespace tinynn
#endif //TINYNN_LAYER_TYPE_H
