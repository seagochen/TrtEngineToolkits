//
// Created by user on 4/22/25.
//

#ifndef INFERENCE_INFER_MODEL_MULTI_H
#define INFERENCE_INFER_MODEL_MULTI_H

#include <string>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "serverlet/trt_engine/trt_engine_multi_ts.h"

template<typename T> class Tensor;  // 假设你的 Tensor 模板在别处声明

struct TensorDefinition {
    std::string name;
    std::vector<int> dims;
};


class InferModelBaseMulti {
public:
    /**
     * @param engine_path    TRT engine 文件路径
     * @param input_name     输入张量名字
     * @param input_dims     输入张量维度 {batch, C, H, W}
     * @param output_names   多个输出张量名字
     * @param output_dims    多个输出张量维度（与 output_names 一一对应）
     */
    InferModelBaseMulti(
            const std::string& engine_path,
            const std::vector<TensorDefinition>& input_defs,
            const std::vector<TensorDefinition>& output_defs);

    virtual ~InferModelBaseMulti();

    /// 子类必须实现：将一张 cv::Mat 预处理到 device buffer
    virtual void preprocess(const cv::Mat& image, int batchIdx) = 0;

    /// 一次性调用推理，支持多个输出
    /// @return true if success
    bool inference();

    /// 拷贝单个输出到 host-vector
    void copyCpuDataFromOutputBuffer(
            const std::string& tensor_name,
            std::vector<float>& output_data,
            int batch_idx = 0);

    /// 拷贝单个输入从 host-vector 到 device buffer
    void copyCpuDataToInputBuffer(
            const std::vector<float>& input_data,
            int batch_idx = 0);

    /// 直接获取输入/输出裸指针（device）
    const float* getOutputBuffer(const std::string& name) const {
        return g_map_trtTensors.at(name).ptr();
    }

protected:
    /// 加载 engine 并创建 context
    bool loadEngine(
            const std::string& engine_path,
            const std::vector<TensorDefinition>& input_defs,
            const std::vector<TensorDefinition>& output_defs);

    /// 为输入和所有输出分配 Tensor buffer
    bool allocateBufsForTrtEngine(
            const TensorDefinition& input_defs,
            const std::vector<TensorDefinition>& output_defs);

    TrtEngineMultiTs*                       g_ptr_engine = nullptr;
    std::string                             g_str_input;
    TensorDefinition                        g_input_defs;
    std::vector<TensorDefinition>           g_output_defs;

    /// key: tensor name, value: device buffer tensor
    std::map<std::string, Tensor<float>>    g_map_trtTensors;

    /// CUDA 流
    cudaStream_t                            g_stream = nullptr;
};

#endif //INFERENCE_INFER_MODEL_MULTI_H
