#ifndef __ENGINE_LOADER_HPP__
#define __ENGINE_LOADER_HPP__

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;


std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)> loadEngine(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file: " << engineFile << std::endl;
        exit(-1);
    }

    file.seekg(0, file.end);
    size_t length = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> data(length);
    file.read(data.data(), length);
    if (!file) {
        std::cerr << "Error reading engine file: " << engineFile << std::endl;
        exit(-1);
    }

    std::unique_ptr<IRuntime, void(*)(IRuntime*)> runtime(createInferRuntime(gLogger), [](IRuntime* r) { delete r; });
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        exit(-1);
    }

    std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)> engine(runtime->deserializeCudaEngine(data.data(), length), [](ICudaEngine* e) { delete e; });
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        exit(-1);
    }

    // Return the engine
    return engine;
}


std::vector<std::string> getTensorNamesFromModel(ICudaEngine* engine) {

    std::vector<std::string> tensor_names;

//    // TensorRT version 10
//    for (int i = 0, e = engine->getNbIOTensors(); i < e; i++) {
//        auto const name = engine->getIOTensorName(i);
//
//        // Add the name to the vector
//        tensor_names.emplace_back(name);
//    }

    // TensorRT version 8
    int nbBindings = engine->getNbBindings();
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine->getBindingName(i);
        tensor_names.emplace_back(name);
    }

    return tensor_names;
}

size_t getTensorSizeByName(ICudaEngine* engine, std::string tensor_name) {
//    // Get the dimensions of the given tensor, TensorRT version 10
//     auto const dims = engine->getTensorShape(tensor_name.c_str());
//
//    // Get the number of dimensions
//    int nbDims = dims.nbDims;
//
//    // Log the number of dimensions for debugging
//    std::cout << "Tensor " << tensor_name << " has " << nbDims << " dimensions." << std::endl;
//
//    // Calculate the size of the tensor by multiplying the dimensions
//    size_t size = 1;
//    for (int i = 0; i < nbDims; ++i) {
//        std::cout << dims.d[i] << "x";
//        size *= dims.d[i];
//    }
//    std::cout << std::endl;
//
//    return size;

    // TensorRT 8
    // 通过名字获取绑定的索引
    int bindingIndex = engine->getBindingIndex(tensor_name.c_str());
    if (bindingIndex == -1) {
        std::cerr << "Tensor name not found: " << tensor_name << std::endl;
        exit(-1);
    }

    // 获取绑定的维度
    auto dims = engine->getBindingDimensions(bindingIndex);

    // 获取维度的数量
    int nbDims = dims.nbDims;

    // 输出维度数量用于调试
    std::cout << "Tensor " << tensor_name << " has " << nbDims << " dimensions." << std::endl;

    // 通过乘积计算张量的大小
    size_t size = 1;
    for (int i = 0; i < nbDims; ++i) {
        std::cout << dims.d[i] << "x";
        size *= dims.d[i];
    }
    std::cout << std::endl;

    return size;
}

std::unique_ptr<IExecutionContext, void(*)(IExecutionContext*)> createExecutionContext(ICudaEngine* engine) {
    if (!engine) {
        std::cerr << "Invalid engine pointer." << std::endl;
        exit(-1);
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        exit(-1);
    }

    return std::unique_ptr<IExecutionContext, void(*)(IExecutionContext*)>(context, [](IExecutionContext* c) { delete c; });
}

#endif // __ENGINE_LOADER_HPP__