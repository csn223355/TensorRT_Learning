/**
 *@file preprocess.cu
 *@author chenshining
 *@version v1.0
 *@date 2024-01-15
 */

#include "preprocess.h"
#include "config.h"
#include "cuda_utils.h"

static u_int8_t *img_buffer_device {nullptr};

struct AffineMatrix
{
    float affine_matrix[6] {0};
};

// 一个线程处理一个像素点
__global__ void preprocess_kernel(uchar *src, float *dst, int dst_width, int dst_height, int edge)
{
    int ix = threadIdx.x + blockIdx.x + blockDim.x;
    int iy = threadIdx.y + blockIdx.y + blockDim.y;
    int idx = ix + iy * dst_width;
    int idx3 = idx * 3;

    if(idx3 >= edge){
        return;
    }
    // normalization（对原图中(x,y)坐标的像素点3个通道进行归一化）
    float c0 = src[idx3] / 255.0f;
    float c1 = src[idx3 + 1] / 255.0f;
    float c2 = src[idx3 + 2] / 255.0f;

    // bgr to rgb
    float temp = c2;
    c2 = c0;
    c0 = temp;

    // NHWC -> NCHW
    // rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float *dst_c0_ptr = dst + ix + dst_width * iy;
    float *dst_c1_ptr = dst_c0_ptr + area;
    float *dst_c2_ptr = dst_c1_ptr + area;
    *dst_c0_ptr = c0;
    *dst_c1_ptr = c1;
    *dst_c2_ptr = c2;
  
}

