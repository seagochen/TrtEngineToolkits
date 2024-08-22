#include "common/engine_loader.hpp"
#include <opencv2/opencv.hpp>
#include <map>

using namespace nvinfer1;

int main() {
    // 使用智能指针加载TensorRT引擎
    auto engine = loadEngine("./models/yolov8n.engine");
    if (!engine) {
        std::cerr << "Failed to load engine." << std::endl;
        return -1;
    }

    // 使用智能指针创建执行上下文
    auto context = createExecutionContext(engine.get());

    // 创建输入输出缓冲区
    std::map<std::string, void*> buffers;

    // 获取模型的所有张量名字
    auto tensor_names = getTensorNamesFromModel(engine.get());

    for (const auto& name : tensor_names) {
        // 获取每个张量的大小
        size_t tensor_size = getTensorSizeByName(engine.get(), name);

        // 分配GPU内存
        void* buffer;
        cudaMalloc(&buffer, tensor_size);

        // 将分配的内存存入map
        buffers[name] = buffer;

        std::cout << "Allocated buffer for tensor: " << name
                  << " with size: " << tensor_size << " bytes." << std::endl;
    }

    // TODO: 将缓冲区传递给推理引擎，并执行推理

    // 清理缓冲区
    for (auto& pair : buffers) {
        cudaFree(pair.second);
    }

    return 0;
}
