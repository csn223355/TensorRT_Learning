# 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置CUDA架构 2060 75
# set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
# 项目信息
project(trt_demo LANGUAGES CXX CUDA)

# 添加CMAKE_MODULE_PATH，否则找不到FindTensorRT.cmake
list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)



# 寻找TensorRT库
find_package(TensorRT REQUIRED)

if (TensorRT_FOUND)
    message(STATUS "Found TensorRT ${TensorRT_VERSION} in ${TensorRT_ROOT_DIR}")
    message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
    message(STATUS "TensorRT include path: ${TensorRT_INCLUDE_DIR}")
    message(STATUS "TensorRT samples path: ${TensorRT_SAMPLE_DIR}")
else()
    message(FATAL_ERROR "Cannot find TensorRT")
endif()

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA include files: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA library directories: ${CUDA_TOOLKIT_ROOT_DIR}/lib64")
else()
    message(FATAL_ERROR "Cannot find CUDA")
endif()

# 添加可执行文件
add_executable(build src/build.cu)
# 头文件
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
# 链接库
target_link_libraries(build PRIVATE ${TensorRT_LIBRARIES})
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")

# 添加可执行文件
add_executable(runtime src/runtime.cu)
# 头文件
target_include_directories(runtime PRIVATE ${TensorRT_INCLUDE_DIR} ${CDUA_INCLUDE_DIRS}) 
# # 链接库
target_link_libraries(runtime PRIVATE ${TensorRT_LIBRARIES})
set_target_properties(runtime PROPERTIES CUDA_ARCHITECTURES "61;70;75")