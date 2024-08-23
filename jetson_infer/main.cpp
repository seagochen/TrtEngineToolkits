#include <opencv2/opencv.hpp>
#include <map>
#include "tensor.h"  // 包含新的 Tensor 类
#include "common/engine_loader.h"


std::map<std::string, CudaTensor> loadTensorsFromModel(nvinfer1::ICudaEngine *engine) {
    // 创建输入输出缓冲区
    std::map<std::string, CudaTensor> buffers;

    // 获取模型的所有张量名字
    auto tensor_names = getTensorNamesFromModel(engine);

    for (const auto& name : tensor_names) {
        // 获取每个张量的大小
        TensorDimensions dims = getTensorDimsByName(engine, name);

        // 创建一个 CudaTensor
        CudaTensor tensor(dims);

        // 保存Tensor到缓冲区
        buffers[name] = std::move(tensor);

        // 输出一些信息
        std::cout << "Allocated buffer for Tensor: " << name
                  << " with size: " << buffers[name].getSize() << " bytes." << std::endl;
    }

    // 返回给上级调用者
    return buffers;
}


int main() {
    // 使用智能指针加载TensorRT引擎
    auto engine = loadEngine("/home//cnn_toolkits.engine");
    if (!engine) {
        std::cerr << "Failed to load engine." << std::endl;
        return -1;
    }

    // 使用智能指针创建执行上下文
    auto context = createExecutionContext(engine.get());

    // 获得与模型有关的Tensors
    auto gpu_tensors = loadTensorsFromModel(engine.get());

    // ビデオを利用し、まずは再生してみて
    cv::VideoCapture cap("highway_tilt_01.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video file." << std::endl;
        return -1;
    }

    return 0;
}
