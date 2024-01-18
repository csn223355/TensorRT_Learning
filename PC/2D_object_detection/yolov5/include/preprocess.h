/**
 *@file preprocess.h
 *@author chenshining
 *@version v1.0
 *@date 2024-01-14
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destory();

void cuda_preprocess(uint8_t *src, int src_width, int src_heigth,
                    float *dest, int dest_width, int dest_height);

void cuda_batch_preprocess(std::vector<cv::Mat> &batch_images, float *dest, int dest_width, int dest_height);

void process_input_gpu(cv::Mat &input_image, float *input_device_buffer);

void process_input_cv_affine(cv::Mat input_image, float1 *input_device_buffer);

void process_input_cpu(cv::Mat &src ,float *input_device_buffer);










