//
// Created by user on 6/18/25.
//

#include <functional> // For std::function


#include "serverlet/models/yolo/infer_yolo_v8_base.h"
#include "serverlet/models/common/yolo_dstruct.h"
#include "serverlet/utils/logger.h"


// infer_yolo_v8.tpp (template implementation file)
template<typename YoloResultType, typename ConvertFunc>
InferYoloV8<YoloResultType, ConvertFunc>::InferYoloV8(
    const std::string& engine_path,
    int maximum_batch,
    int maximum_items,
    int infer_features_val,
    const std::vector<TensorDefinition>& output_tensor_defs,
    ConvertFunc converter)
    : InferModelBaseMulti(engine_path,
                          {{"images", {maximum_batch, 3, 640, 640}}},
                          output_tensor_defs),
      maximum_batch(maximum_batch),
      maximum_items(maximum_items),
      image_width(640),
      image_height(640),
      image_channels(3),
      infer_features(infer_features_val),
      infer_samples(8400),
      m_converter(converter)
{
    g_vec_output.resize(infer_features * infer_samples, 0.0f);
}

// ... destructor and preprocess (mostly unchanged, perhaps generalize log tag)

template<typename YoloResultType, typename ConvertFunc>
std::vector<YoloResultType> InferYoloV8<YoloResultType, ConvertFunc>::postprocess(const int batchIdx, const float cls, const float iou) {
    if (batchIdx >= maximum_batch) {
        LOG_ERROR("InferYoloV8", "batchIdx >= maximum_batch"); // More generic log
        return {};
    }

    const float* cuda_device_ptr = accessCudaBufByBatchIdx("output0", batchIdx);

    int results = sct_yolo_post_proc(cuda_device_ptr, g_vec_output, infer_features, infer_samples, cls, (std::is_same_v<YoloResultType, YoloPose>)); // Pass boolean for pose
    if (results < 0) {
        LOG_ERROR("InferYoloV8", "No results found after post-processing");
        return {};
    }

    std::vector<YoloResultType> yolo_results;
    m_converter(g_vec_output, yolo_results, infer_features, results); // Use the template function object

    // Apply NMS if desired (can be a template parameter or conditional)
    if constexpr (std::is_same_v<YoloResultType, YoloPose>) { // Example of conditional NMS
        yolo_results = nms(yolo_results, iou);
    }
    return yolo_results;
}