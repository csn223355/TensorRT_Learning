/**
 *@file build.cu
 *@author chenshining
 *@version v1.0
 *@date 2024-01-14
 */
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

#include <NvInfer.h>
#include <NvOnnxParaser.h>

// 定义校准数据读取器

class CalibrationDataLoader : public nvinfer1::IInt8Calibrator
{

private:
    std::string mDataDir;
    std::string mCacheFileName;
    std::vector<std::string> mFileNames;
    int mBatchSize;
    nvinfer1::Dims mIntputDims;
    int mIntputCount;
    float *mDeviceBatchData{nullptr};
    int mBatchCount;
    int mImgSize;
    int mCurBatch{0};
    std::vector<char> mCalibrationCache;

public:
    CalibrationDataLoader()
    {

// TODO
    }

}







// test
int main(){


  return 0;
}