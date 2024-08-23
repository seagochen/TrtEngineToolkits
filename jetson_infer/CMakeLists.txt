cmake_minimum_required(VERSION 3.14)
project(jetson_infer)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 统一设置 CUDA 路径
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBRARIES /usr/local/cuda/lib64/libcudart.so)

# 检查系统架构以区分 TensorRT 的设置
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")  # Jetson 平台（ARM64 架构）
    message(STATUS "Configuring for Jetson platform (ARM64)")

    # 使用系统路径查找 TensorRT
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS /usr/include /usr/local/include
    )

    find_library(NVINFER_LIB nvinfer
            HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib
    )

    find_library(NVPARSERS_LIB nvparsers
            HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib
    )

    find_library(NVONNXPARSER_LIB nvonnxparser
            HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib
    )

    set(TENSORRT_LIBS
            ${NVINFER_LIB}
            ${NVPARSERS_LIB}
            ${NVONNXPARSER_LIB}
    )

else()  # x86 平台
    message(STATUS "Configuring for x86 platform")

    # 设置TensorRT的路径, 使用相对路径
    set(TENSORRT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/tensorrt/include)
    set(TENSORRT_LIBRARIES ${PROJECT_SOURCE_DIR}/tensorrt/lib)
    set(TENSORRT_LIBS
            ${TENSORRT_LIBRARIES}/libnvinfer.so
            ${TENSORRT_LIBRARIES}/libnvonnxparser.so
    )
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 指定源文件, common, utils
file(GLOB SOURCE_FILES "common/*.cpp" "utils/*.cpp")

# 添加可执行文件
add_executable(jetson_infer main.cpp ${SOURCE_FILES})

# 包含目录设置
target_include_directories(jetson_infer PRIVATE
        ${PROJECT_SOURCE_DIR}/common
        ${PROJECT_SOURCE_DIR}/utils
        ${PROJECT_SOURCE_DIR}/cumatrix
        ${PROJECT_SOURCE_DIR}/cnn_toolkits
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIRS}  # 注意这里使用的是架构区分后的路径
        ${OpenCV_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(jetson_infer PRIVATE
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TENSORRT_LIBS}  # 同样，这里根据架构区分后的库进行链接
        ${PROJECT_SOURCE_DIR}/cumatrix/libcumatrix.a  # Integrate CUDA component: cumatrix
        ${PROJECT_SOURCE_DIR}/cnn_toolkits/libcnn_toolkits.a  # Integrate CUDA component: cnn_toolkits
)