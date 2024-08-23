#include "engine_loader.h"
#include <string>
#include <fstream>
#include <iostream>

#if NV_TENSORRT_MAJOR >= 10
#define TENSORRT_VERSION_10
#elif NV_TENSORRT_MAJOR == 8
#define TENSORRT_VERSION_8
#else
#error "Unsupported TensorRT version"
#endif

using namespace nvinfer1;


// Logger class for inference engine
class Logger : public nvinfer1::ILogger {
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

    file.seekg(0, std::ifstream::end);
    size_t length = file.tellg();
    file.seekg(0, std::ifstream::beg);

    if (length > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        std::cerr << "File size is too large to be read into memory." << std::endl;
        exit(-1);
    }

    std::vector<char> data(static_cast<std::streamsize>(length));
    file.read(data.data(), static_cast<std::streamsize>(length));
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
    for (int i = 0, e = engine->getNbIOTensors(); i < e; i++) {
        auto const name = engine->getIOTensorName(i);
        tensor_names.emplace_back(name);
    }
#elif defined(TENSORRT_VERSION_8)
    int nbBindings = engine->getNbBindings();
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine->getBindingName(i);
        tensor_names.emplace_back(name);
    }
#endif

    return tensor_names;
}


TensorDimensions getTensorDimsByName(ICudaEngine* engine, const std::string& tensor_name, tensor_type type) {
    TensorDimensions tensor_dims;

#ifdef TENSORRT_VERSION_10
    auto const dims = engine->getTensorShape(tensor_name.c_str());
    int nbDims = dims.nbDims;

    std::vector<int> dim_sizes;
    for (int i = 0; i < nbDims; ++i) {
        dim_sizes.push_back(dims.d[i]);
    }

    tensor_dims = TensorDimensions(dim_sizes, type);  // Assume FLOAT32, adjust as necessary

#elif defined(TENSORRT_VERSION_8)
    int bindingIndex = engine->getBindingIndex(tensor_name.c_str());
    if (bindingIndex == -1) {
        std::cerr << "Tensor name not found: " << tensor_name << std::endl;
        exit(-1);
    }

    auto dims = engine->getBindingDimensions(bindingIndex);
    int nbDims = dims.nbDims;

    std::vector<int> dim_sizes;
    dim_sizes.reserve(nbDims);

    for (int i = 0; i < nbDims; ++i) {
        dim_sizes.push_back(dims.d[i]);
    }

    tensor_dims = TensorDimensions(dim_sizes, type);  // Assume FLOAT32, adjust as necessary

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

    return {context, [](IExecutionContext* c) { delete c; }};
}