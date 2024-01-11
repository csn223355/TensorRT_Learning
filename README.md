# TensorRT_demo

## TensorRT_demo目录结构如下：
```
.
├── build
│   └── CMakeFiles
├── cmake
├── src
└── weights
```
`build` 存放编译生成文件，`cmake`存放cmake模块文件，`src`存放源码，`weights`生成的模型文件如*.engine等。

## demo介绍
这个demo的主要演示了如何使用TensorRT构建engine并保存到文件中，以及如后反序列化engine文件，推理模型。
定义了一个简单的`MLP`层，作为模型，用列向量`[2,4,8]`作为输入，最后输出一个2 * 1 的列向量。 



