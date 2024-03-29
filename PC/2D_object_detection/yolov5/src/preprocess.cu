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
    float value[6] {0};
};

// 一个线程处理一个像素点
__global__ void preprocess_kernel(uchar *src, float *dst, int dst_width, int dst_height, int edge)
{
    int ix = threadIdx.x + blockIdx.x + blockDim.x;
    int iy = threadIdx.y + blockIdx.y + blockDim.y;
    int idx = ix + iy * dst_width;
    int idx3 = idx * 3;

    if(idx >= edge){
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

__global__ void warpaffine_kernel(uchar *src, float *dst, int src_line_size, int src_width, int src_height, 
    int dst_width, int dst_height, uchar const_value_st, AffineMatrix d2s, int edge)
{
    int dx = threadIdx.x + blockDim.x * blockIdx.x;
    int dy = threadIdx.y + blockDim.y * blockIdx.y;
    int id = dx + dy * dst_width;
    int id3 = id * 3;
    if(id >= edge){
        return;
    }

    // 从d2s中读取变换矩阵
    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    // 计算仿射变换后的源图像坐标：
    // 计算给定目标图像坐标 (dx, dy) 在源图像空间中的变换坐标 (src_x, src_y)。
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;   
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;

    float c0, c1, c2;
    if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
        // 超出边界的像素点用const_value_st填充
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }else{
        // 双线性插值，实现图像的放大缩小

        // 计算插值位置 的 左上角（x_low, y_low），右下角（x_high, y_high）
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uchar const_value[] = {const_value_st, const_value_st, const_value_st};

        // 计算 插值位置 与 最近四个像素的距离
        float ly = src_y - y_low; // 垂直方向 插值位置 与 左上角像素点的距离
        float lx = src_x - x_low; // 水平方向 插值位置 与 左上角像素点的距离
        float hy = 1 - ly; // 垂直方向 补数
        float hx = 1 - lx; // 水平方向 补数
        // w1
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t *v1 = const_value;
        uint8_t *v2 = const_value;
        uint8_t *v3 = const_value;
        uint8_t *v4 = const_value;

        if (y_low >= 0){
            if (x_low >= 0){
                v1 = src + y_low * src_line_size + x_low * 3; //v1 被赋值为源图像上左上角像素点 (x_low, y_low) 的颜色值。
            }

            if (x_high < src_width){
                v2 = src + y_low * src_line_size + x_high * 3; //v2 被赋值为源图像上右上角像素点 (x_high, y_low) 的颜色值
            }
        }

        if (y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3; //v3 被赋值为源图像上左下角像素点 (x_low, y_high) 的颜色值

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3; //v4 被赋值为源图像上右下角像素点 (x_high, y_high) 的颜色值
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
  }

    // bgr to rgb
    float t = c2;
    c2 = c0;
    c0 = t;

    // normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float *pdst_c0 = dst + dy * dst_width + dx;
    float *pdst_c1 = pdst_c0 + area;
    float *pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;

}

// GPU 作归一化 、 BGR2RGB、NHWC to NCHW
void cuda_pure_preprocess(uchar *src, float *dst, int dst_width, int dst_height)
{
    int img_size = dst_width * dst_height * 3;
    CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice));
    dim3 blockSize{16,16};
    dim3 gridSize{(dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y};
    preprocess_kernel<<<gridSize, blockSize>>>(img_buffer_device, dst, dst_width, dst_height, dst_width * dst_height);
}

void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height)
{

    int img_size = src_width * src_height * 3;
    CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice));

    // 计算变换矩阵
    AffineMatrix s2d, d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    // 一个线程处理一个像素点，一共需要 dst_height * dst_width 个线程
    int jobs = dst_height * dst_width;
    dim3 blockSize{16,16};
    dim3 gridSize{(dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y};
    // 调用kernel函数

    warpaffine_kernel<<<gridSize, blockSize>>>(img_buffer_device, 
                                        dst, src_width * 3, src_width,
                                        src_height, dst_width,
                                        dst_height, 128, d2s, jobs);
    }

