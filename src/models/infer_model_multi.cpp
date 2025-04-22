//
// Created by user on 4/22/25.
//

#include "simple_cuda_toolkits/tensor_utils.hpp"
#include "serverlet/utils/logger.h"
#include "serverlet/models/infer_model_multi.h"
#include <numeric>
#include <cuda_runtime.h>


InferModelBaseMulti::InferModelBaseMulti(
        const std::string& engine_path,
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs)
        : g_str_input(input_defs[0].name)
{
    // 初始化TrtEngine
    g_ptr_engine = new TrtEngineMultiTs();

    // 先创建一个CUDA流
    cudaError_t err = cudaStreamCreate(&g_stream);
    if (err != cudaSuccess) {
        auto err_str = std::string(cudaGetErrorString(err));
        LOG_ERROR("InferModelBase", "Failed to create CUDA stream: " + err_str);
    }

    // 加载引擎
    if (!loadEngine(engine_path, input_defs, output_defs)) {
        LOG_ERROR("InferModelBase", "Failed to load engine: " + engine_path);
        exit(EXIT_FAILURE);
    }
    LOG_VERBOSE("InferModelBase", "Engine loaded from: " + engine_path);

    // 分配输入输出 buffer
    if (!allocateBufsForTrtEngine(input_defs, output_defs)) {
        LOG_ERROR("InferModelBase", "Failed to allocate buffers");
        exit(EXIT_FAILURE);
    }
    LOG_VERBOSE("InferModelBase", "Buffers allocated");
}

InferModelBaseMulti::~InferModelBaseMulti() {

    // 销毁引擎
    delete g_ptr_engine;
    g_map_trtTensors.clear();

    // 销毁流
    cudaStreamDestroy(g_stream);
}

bool InferModelBaseMulti::loadEngine(
        const std::string& engine_path,
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs)
{
    if (!g_ptr_engine->loadFromFile(engine_path)) {
        LOG_ERROR("loadEngine", "loadFromFile failed: " + engine_path);
        return false;
    }

    // 绑定输入
    std::vector<std::string> input_names;
    std::vector<nvinfer1::Dims4> input_dims;
    for (auto& input_def : input_defs) {
        const auto& name = input_def.name;
        const auto& dims = input_def.dims;

        // 检查输入维度
        if (dims.size() != 4) {
            LOG_ERROR("loadEngine", "input_dims 必须为 4 维");
            return false;
        }

        // 设置输入维度
        nvinfer1::Dims4 dims4{
                dims[0],
                dims[1],
                dims[2],
                dims[3]
        };

        // 将输入名称和维度添加到列表中
        input_names.push_back(name);
        input_dims.push_back(dims4);
    }

    // 绑定输出名字
    std::vector<std::string> output_names;
    for (auto& output_def : output_defs) {
        const auto& name = output_def.name;
        const auto& dims = output_def.dims;

        // 将输出名称和维度添加到列表中
        output_names.push_back(name);
    }

    // 将输入和输出名称和维度传递给引擎
    if (!g_ptr_engine->createContext(input_names, input_dims, output_names)) {
        LOG_ERROR("loadEngine", "createContext 失败");
        return false;
    }

    return true;
}


bool InferModelBaseMulti::allocateBufsForTrtEngine(const TensorDefinition& input_defs,
                                                   const std::vector<TensorDefinition>& output_defs)
{
    try {
        // 输入 buffer
        g_map_trtTensors[input_defs.name] = createZerosTensor<TensorType::FLOAT32>(input_defs.dims);

        // 为每个输出分配 buffer
        for (auto& output_def : output_defs) {
            const auto& name = output_def.name;
            const auto& dims = output_def.dims;
            g_map_trtTensors[name] = createZerosTensor<TensorType::FLOAT32>(dims);
        }

        // 检查是否分配成功
        if (g_map_trtTensors.empty()) {
            LOG_ERROR("allocateBufsForTrtEngine", "No tensors allocated");
            return false;
        } else {
            return true;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("allocateBufsForTrtEngine", e.what());
        return false;
    }
}


bool InferModelBaseMulti::inference() {
//    // 构造输出列表
//    std::vector<Tensor<float>> outputs;
//    outputs.reserve(g_vec_outputNames.size());
//    for (auto& name : g_vec_outputNames) {
//        outputs.push_back(g_map_trtTensors[name]);
//    }
//    // 调用 TRT infer (假设支持多输出重载)
//    return g_ptr_engine->infer(
//            g_map_trtTensors[g_str_input],
//            outputs);

    // 构造输出列表
    std::vector<Tensor<float>> outputs;
    outputs.reserve(g_vec_outputNames.size());

    return g_ptr_engine->infer(g_map_trtTensors[g_str_input], )


}

void InferModelBaseMulti::copyCpuDataFromOutputBuffer(
        const std::string& tensor_name,
        std::vector<float>& output_data,
        int batch_idx)
{
    auto& tensor = g_map_trtTensors.at(tensor_name);
    // 计算单样本大小（不含 batch 维）
    int single = std::accumulate(
            tensor.dims().begin()+1,
            tensor.dims().end(), 1,
            std::multiplies<int>());
    if ((int)output_data.size() != single) {
        throw std::runtime_error(
                "Output data size mismatch for " + tensor_name);
    }
    size_t offset = batch_idx * single;
    cudaMemcpy(
            output_data.data(),
            tensor.ptr() + offset,
            sizeof(float) * single,
            cudaMemcpyDeviceToHost);
}

void InferModelBaseMulti::copyCpuDataToInputBuffer(
        const std::vector<float>& input_data,
        int batch_idx)
{
    auto& tensor = g_map_trtTensors.at(g_str_input);
    int single = std::accumulate(
            g_vec_inputDims.begin()+1,
            g_vec_inputDims.end(), 1,
            std::multiplies<int>());
    if ((int)input_data.size() != single) {
        throw std::runtime_error("Input data size mismatch");
    }
    size_t offset = batch_idx * single;
    cudaMemcpy(
            tensor.ptr() + offset,
            input_data.data(),
            sizeof(float) * single,
            cudaMemcpyHostToDevice);
}