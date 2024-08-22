cmake_minimum_required(VERSION 3.14)
project(jetson_infer)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# 设置CUDA路径
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBRARIES /usr/local/cuda/lib64)

# 设置TensorRT的路径, 使用相对路径 ./tensorrt
set(TENSORRT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/tensorrt/include)
set(TENSORRT_LIBRARIES ${PROJECT_SOURCE_DIR}/tensorrt/lib)

# 添加链接器库路径
link_directories(${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES})

# 使用 pkg-config 查找 OpenCV
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)

# 添加 common 文件夹路径和 CUDA, OpenCV, TensorRT 头文件路径
include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories(${PROJECT_SOURCE_DIR}/utils)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIRS})

# 指定源文件, common, utils
file(GLOB SOURCE_FILES "common/*.cpp" "utils/*.cpp")

# 添加可执行文件
add_executable(jetson_infer main.cpp ${SOURCE_FILES})

# 链接CUDA、TensorRT和OpenCV库
target_link_libraries(jetson_infer
        cudart
        ${OpenCV_LIBRARIES}
        ${TENSORRT_LIBRARIES}/libnvinfer.so
        ${TENSORRT_LIBRARIES}/libnvonnxparser.so
)
