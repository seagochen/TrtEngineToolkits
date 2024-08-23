//
// Created by vipuser on 8/23/24.
//

#ifndef JETSON_INFER_ENGINE_LOADER_H
#define JETSON_INFER_ENGINE_LOADER_H

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <vector>
#include <memory>
#include <iostream>

#include "tensor_utils.h"

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

/**
 * @brief Loads a serialized TensorRT engine from a file.
 *
 * This function reads a serialized engine from the specified file and
 * deserializes it to create an ICudaEngine object, which is used to execute
 * inference on the GPU.
 *
 * @param engineFile The path to the engine file.
 * @return A unique pointer to the deserialized ICudaEngine, with a custom deleter
 *         to ensure proper cleanup.
 */
std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> loadEngine(const std::string& engineFile);

/**
 * @brief Retrieves the names of all tensors in the model.
 *
 * This function extracts the names of all input and output tensors in the
 * TensorRT engine. The function supports both TensorRT 8 and TensorRT 10 APIs.
 *
 * @param engine A pointer to the ICudaEngine object.
 * @return A vector of strings containing the names of all tensors in the model.
 */
std::vector<std::string> getTensorNamesFromModel(nvinfer1::ICudaEngine* engine);

/**
 * @brief Retrieves the dimensions and size of a specific tensor by name.
 *
 * This function queries the dimensions of the specified tensor in the engine
 * and calculates its size in bytes. The function returns a TensorDimensions
 * struct containing the dimensions and size. The function supports both
 * TensorRT 8 and TensorRT 10 APIs.
 *
 * @param engine A pointer to the ICudaEngine object.
 * @param tensor_name The name of the tensor whose dimensions are to be retrieved.
 * @return A TensorDimensions struct containing the tensor's dimensions and size in bytes.
 */
TensorDimensions getTensorDimsByName(nvinfer1::ICudaEngine* engine, const std::string& tensor_name);

/**
 * @brief Creates an execution context for the given TensorRT engine.
 *
 * This function creates an IExecutionContext object from the given ICudaEngine.
 * The execution context is used to execute inference. The function returns a
 * unique pointer to the IExecutionContext with a custom deleter to ensure proper cleanup.
 *
 * @param engine A pointer to the ICudaEngine object.
 * @return A unique pointer to the created IExecutionContext object.
 */
std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> createExecutionContext(nvinfer1::ICudaEngine* engine);

#endif //JETSON_INFER_ENGINE_LOADER_H
