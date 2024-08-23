#include "engine_loader.h"

#include <string>
#include <fstream>
#include <iostream>


// Determine the TensorRT version
#if NV_TENSORRT_MAJOR >= 10
#define TENSORRT_VERSION_10
#elif NV_TENSORRT_MAJOR == 8
#define TENSORRT_VERSION_8
#else
#error "Unsupported TensorRT version"
#endif

using namespace nvinfer1;

// Define a global logger here.
Logger gLogger;

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

    return engine;
}

std::vector<std::string> getTensorNamesFromModel(ICudaEngine* engine) {
    std::vector<std::string> tensor_names;

#ifdef TENSORRT_VERSION_10
    // TensorRT version 10
    for (int i = 0, e = engine->getNbIOTensors(); i < e; i++) {
        auto const name = engine->getIOTensorName(i);
        tensor_names.emplace_back(name);
    }
#elif defined(TENSORRT_VERSION_8)
    // TensorRT version 8
    int nbBindings = engine->getNbBindings();
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine->getBindingName(i);
        tensor_names.emplace_back(name);
    }
#endif

    return tensor_names;
}

// Function to get tensor dimensions and size
TensorDimensions getTensorDimsByName(ICudaEngine* engine, const std::string& tensor_name) {
    TensorDimensions tensor_dims;

#ifdef TENSORRT_VERSION_10
    // TensorRT version 10
    auto const dims = engine->getTensorShape(tensor_name.c_str());
    int nbDims = dims.nbDims;

    tensor_dims.size = 1;
    for (int i = 0; i < nbDims; ++i) {
        tensor_dims.dims.push_back(dims.d[i]);
        tensor_dims.size *= dims.d[i];
    }

#elif defined(TENSORRT_VERSION_8)
    // TensorRT version 8
    int bindingIndex = engine->getBindingIndex(tensor_name.c_str());
    if (bindingIndex == -1) {
        std::cerr << "Tensor name not found: " << tensor_name << std::endl;
        exit(-1);
    }

    auto dims = engine->getBindingDimensions(bindingIndex);
    int nbDims = dims.nbDims;

    tensor_dims.size = 1;
    for (int i = 0; i < nbDims; ++i) {
        tensor_dims.dims.push_back(dims.d[i]);
        tensor_dims.size *= dims.d[i];
    }

#endif

    return tensor_dims;
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

