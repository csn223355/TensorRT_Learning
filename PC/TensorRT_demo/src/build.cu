/**
 *@file bulid.cu
 *@author chenshining
 *@version v1.0
 *@date 2024-01-02
 */

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <fstream>
#include <memory>
#include <NvInfer.h>

/**
 * 1. 构建builder
 * 2. 创建网络定义：builder ---> network
 * 3. 配置参数：builder ---> config
 * 4. 生成engine：builder ---> engine (network, config)
 * 5. 序列化保存：engine ---> serialize
 * 6. 释放资源：delete
*/

// 构建logger 管理日志打印的级别
// TRTLogger 继承 nvinfer1::ILogger
class TRTLogger : public nvinfer1::ILogger{
public:
    void log(Severity severity, const char *mesg) noexcept override{
        //  屏蔽 INFO 输出
        if(severity != Severity::kINFO){
            std::cout << mesg << std::endl; 
        }
    }

};

// 保存权重为文件
void saveWeights(const std::string &file_name,const float *data,int size){
    std::ofstream out_file(file_name,std::ios::binary);
    assert(out_file.is_open() && "save weights failed");
    out_file.write((char *)(&size),sizeof(int));        // 保存 权重的大小
    out_file.write((char *)(data),size * sizeof(float));// 保存 权重的数据
    out_file.close();
}

// 读取权重文件
std::vector<float> loadWeights(const std::string &file_name){
    std::ifstream read_file(file_name,std::ios::binary);
    assert(read_file.is_open() && "load weights failed");
    int size {0};
    read_file.read((char *)(&size),sizeof(int));                               // 读取权重的大小
    std::vector<float> data(size);
    read_file.read((char *)(data.data()),size * sizeof(float));
    read_file.close();
    return data;

}


int main(){

    // 1. ============创建logger==============
    TRTLogger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    // 2. ============创建网络定义：builder ---> network==============
    // 显性Batch
    // 1 << 0 = 1，二进制移位，左移0位，相当于1（y左移x位，相当于y乘以2的x次方）
    auto explict_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 调用createNetworkV2创建网络定义，参数是显性batch
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explict_batch));
    
    // 定义网络结构
    // mlp多层感知机：input(1,3,1,1) --> fc1 --> sigmoid --> output (2)

    // 创建一个input tensor ，参数分别是：name, data type, dims
    const int input_size{3}, output_size(2);
    nvinfer1::ITensor *input = network->addInput("data",nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(explict_batch, input_size, 1, 1));
    
    // 创建全连接层fc1
    // weight and bias
    const float *fc1_weights_data {new float[input_size * output_size] {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}};
    const float *fc1_bias_data {new float[output_size] {0.1,0.5}};

    // 将权重保存到文件中，演示从别的来源加载权重
    saveWeights("weights/fc1.wts",fc1_weights_data, input_size * output_size);
    saveWeights("weights/fc1.bias",fc1_bias_data, output_size);

    // 读取权重
    auto fc1_weights_vec = loadWeights("weights/fc1.wts");
    auto fc1_bias_vec = loadWeights("weights/fc1.bias");
    // 转为nvinfer1::Weights类型，参数分别是：data type, data, size
    nvinfer1::Weights fc1_weight{nvinfer1::DataType::kFLOAT, fc1_weights_vec.data(), fc1_weights_vec.size()};
    nvinfer1::Weights fc1_bias{nvinfer1::DataType::kFLOAT, fc1_bias_vec.data(), fc1_bias_vec.size()};

    // 调用addFullyConnected创建全连接层，参数分别是：input tensor, output size, weight, bias
    nvinfer1::IFullyConnectedLayer *fc1 = network->addFullyConnected(*input, output_size,fc1_weight, fc1_bias);
    // 添加sigmoid激活层，参数分别是：input tensor, activation type（激活函数类型）
    nvinfer1::IActivationLayer *sigmoid  = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

    // 设置输出名字
    sigmoid->getOutput(0)->setName("output");
    // 标记输出，没有标记会被当成顺时针优化掉
    network->markOutput(*sigmoid->getOutput(0));

    // 设定最大batch size
    builder->setMaxBatchSize(explict_batch);

    // 3.  ======配置参数：builder ---> config ======
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig> (builder->createBuilderConfig());
    // 设置最大工作空间大小，单位是字节
    config->setMaxWorkspaceSize(1 << 28); // 256MiB

    // 4. ====== 创建engine：builder ---> network ---> config ======
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network,*config));
    if(!engine){
        std::cerr << "Failed to create engine!" << std::endl;
        return -1;
    }

    // 5.====== 序列化engine ======
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(engine->serialize());
    // 存入文件
    std::ofstream output_file("weights/mlp.engine",std::ios::binary);
    assert(output_file.is_open() && "Failed to open file for writing engine");
    output_file.write((char *)serialized_engine->data(),serialized_engine->size());
    output_file.close();
    
    //6. ====== 释放资源 ======
    delete fc1_weights_data;
    fc1_weights_data = nullptr;
    delete fc1_bias_data;
    fc1_bias_data = nullptr;
    
    std::cout << "engine文件生成成功！" << std::endl;
    return 0;
}





