/**
 *@file cuda_utils.h
 *@author chenshining
 *@version v1.0
 *@date 2024-01-15
 */

#pragma once

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if(error_code != cudaSuccess){ \
            std::cerr << "CUDA error" << error_code << " at " << __FILE__ << " : " << __LINE__ ;\
            assert(0);\

        }\
    }
#endif     // CUDA_CHECK
