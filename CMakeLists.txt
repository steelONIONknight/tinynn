cmake_minimum_required(VERSION 3.10)
project(tinynn)

enable_language(CUDA)


#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/layer)

#必须加上以下两条语句，不然clion识别cuda函数
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

#option(TINYNN_CUDA ON)
set(CMAKE_CXX_STANDARD 11)


add_subdirectory(layer/cuda)
set(EXTRA_LIBS ${EXTRA_LIBS} CudaOperator)


if (TINYNN_CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

    CUDA_ADD_LIBRARY(tinynn_cuda_lib STATIC "${tinynn_SRCS_CUDA}")
    add_compile_definitions(TINYNN_CUDA = 1)
    set_target_properties(tinynn_cuda_lib
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            LINKER_LANGUAGE CXX)
    target_include_directories(tinynn_cuda_lib
            PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include/tinynn>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/layer>)

    target_link_libraries(tinynn PUBLIC "${CUDA_LIBRARIES}")
    target_link_libraries(tinynn PRIVATE "tinynn_cuda_lib")
endif ()


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

add_executable(tinynn main.cpp
        mat.h mat.cpp
        allocator.h allocator.cpp
        datareader.h datareader.cpp
        paramdict.h paramdict.cpp
        modelbin.h modelbin.cpp
        c_api.cpp c_api.h
        option.cpp option.h
        cpu.h cpu.cpp
        layer.cpp layer.h
        gpu.h gpu.cpp
        layer/absval.cpp layer/absval.h
        layer/cast.cpp layer/cast.h
        layer/cuda/absval_cuda.h layer/cuda/absval_cuda.cpp
        layer_declaration.h
        tests/prng.h tests/testutil.h
        tests/test_absval.cpp tests/test_absval.h
        layer/innerproduct.cpp layer/innerproduct.h
        tests/test_innerproduct.cpp tests/test_innerproduct.h
        layer/convolution.cpp layer/convolution.h
        layer_type.h layer_type_enum.h
        layer/padding.cpp layer/padding.h)


target_link_libraries(tinynn ${EXTRA_LIBS}
        ${CUDA_INCLUDE_DIRS})