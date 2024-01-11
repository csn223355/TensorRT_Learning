/**
 *@file runtime.cu
 *@author chenshining
 *@version v1.0
 *@date 2024-01-05
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

#include <memory>
#include "cuda_runtime.h"
#include "NvInfer.h"


/*
使用.cu是希望使用CUDA的编译器NVCC，会自动连接cuda库

TensorRT runtime 推理过程

1. 创建一个runtime对象
2. 反序列化生成engine：runtime ---> engine
3. 创建一个执行上下文ExecutionContext：engine ---> context

    4. 填充数据
    5. 执行推理：context ---> enqueueV2

6. 释放资源：delete

*/

class TRTLogger : public nvinfer1::ILogger{
public:
    void log(Severity severity, const char *mesg) noexcept override{
        //  屏蔽 INFO 输出
        if(severity != Severity::kINFO){
            std::cout << mesg << std::endl; 
        }
    }

};

// 加载模型
std::vector<unsigned char> loadEngineModel(const std::string &file_name){
    std::ifstream file(file_name,std::ios::binary);
    assert(file.is_open() && "fail to load engine file");
    // 移动读取指针位置到文件末尾
    file.seekg(0,std::ios::end);
    // 获取当前读取指针位置，即文件的大小
    size_t size = file.tellg();
    // 将读取指针的位置 移动到文件开始的地方
    file.seekg(0,std::ios::beg);
    std::vector<unsigned char> data(size);
    file.read((char *)data.data(),size);
    file.close();
    return data;

}

int main(){
    // 1. 创建 runtime 对象
    TRTLogger logger;

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    
    // 2. 反序列化生成 engine
    auto engineModel = loadEngineModel("./weights/mlp.engine");
    // 调用runtime的反序列化方法，生成engine，参数分别是：模型数据地址，模型大小，pluginFactory
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineModel.data(), engineModel.size(), nullptr));
    
    if(!engine){
        std::cerr << "fail to deserialise engine" << std::endl;
        return -1;
    }
    // 3. 创建一个上下文
    auto context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // 4. 填充数据
    // 设置stream流
    cudaStream_t stream{nullptr};
    cudaStreamCreate(&stream);

    // 数据流转：host --> device ---> inference ---> host
    // 输入数据
    
    float *host_input_data { new float[3]{2,4,8} };
    
    int input_data_size { 3 * sizeof(float) };
    float *device_input_data {nullptr};

    // 输出数据
    
    float *host_output_data  { new float[2]{0,0} };
    
    int output_data_size { 2 * sizeof(float) };
    float *device_output_data {nullptr};

    // 申请device内存
    cudaMalloc((void **)&device_input_data, input_data_size);
    cudaMalloc((void **)&device_output_data, output_data_size);

    // host --> device
    // 参数分别是：目标地址，源地址，数据大小，拷贝方向
    cudaMemcpyAsync(device_input_data, host_input_data, input_data_size, cudaMemcpyHostToDevice, stream);

    // bindings 告诉 context 输入输出数据的位置
    float *bindings[] = {device_input_data, device_output_data};
    // 5. 执行推理
    bool success {context->enqueueV2((void **) bindings, stream, nullptr)};

    if(!success){
        std::cerr << "推理失败" << std::endl;
        return -1;
    }

    // 拷贝数据 device -> host
    cudaMemcpyAsync(host_output_data, device_output_data, output_data_size, cudaMemcpyDeviceToHost, stream);
    // 等待流执行完毕
    cudaStreamSynchronize(stream);
    // 输出结果
    std::cout << "输出结果：" << host_output_data[0] << " " << host_output_data[1] << std::endl;
    
    // 释放资源
    cudaStreamDestroy(stream);
    cudaFree(device_input_data);
    device_input_data = nullptr;
    cudaFree(device_output_data);
    device_output_data = nullptr;
    
    delete[] host_input_data;
    host_input_data = nullptr;
    delete[] host_output_data;
    host_output_data = nullptr;

    return 0;


    
}
