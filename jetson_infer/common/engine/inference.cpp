#include <simple_cuda_toolkits/vision/colorspace.h>
#include <simple_cuda_toolkits/vision/normalization.h>
#include <simple_cuda_toolkits/tsutils/permute_3D.h>
#include <simple_cuda_toolkits/yolo8utils/yolov8.h>
#include <simple_cuda_toolkits/tensor.hpp>

#include "common/engine/inference.h"

#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640
#define INPUT_CHANNELS 3

#define OUTPUT_SAMPLES 8400
#define OUTPUT_OBJECTS 84


// Initialize some global variables
Tensor<float> g_input_temp = createZerosTensor<TensorType::FLOAT32>(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS);
Tensor<float> g_output_temp = createZerosTensor<TensorType::FLOAT32>(OUTPUT_SAMPLES, OUTPUT_OBJECTS);
cv::Mat g_resized =  cv::Mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3);
cv::Mat g_floated = cv::Mat(INPUT_HEIGHT, INPUT_WIDTH, CV_32FC3);


void preprocess(cv::Mat &image, Tensor<float> &output) {
    // Resize the image to the target size
    cv::resize(image, g_resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

    // Convert the image to float32
    g_resized.convertTo(g_floated, CV_32FC3);

    // Copy the image to the tensor
    cudaMemcpy(g_input_temp.ptr(), g_floated.data, g_floated.total() * g_floated.elemSize(),
               cudaMemcpyHostToDevice);

    // Convert the color space from BGR to RGB
    sctBGR2RGB(g_input_temp.ptr(), output.ptr(),
               INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);

    // Normalize the image data
    sctNormalizeData(output.ptr(), g_input_temp.ptr(),
                     INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);

    // Permute the image data
    sctPermute3D(g_input_temp.ptr(), output.ptr(),
                 INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS,
                 2, 0, 1);
}


// Postprocess helper to extract box and confidence information
YoloResult extractYoloResult(const std::vector<float> &data, int index, int features) {
    YoloResult result;
    result.lx = static_cast<int>(data[index * features]);
    result.ly = static_cast<int>(data[index * features + 1]);
    result.rx = static_cast<int>(data[index * features + 2]);
    result.ry = static_cast<int>(data[index * features + 3]);
    result.cls = static_cast<int>(data[index * features + 5]);
    result.conf = data[index * features + 4];
    return result;
}


//// Pose postprocessing with keypoints
//void pose_postprocess(Tensor<float> &output, std::vector<YoloResult> &results, float confidence {
//
//    sctYolov8PosePostProcessing(output.ptr(), g_gpu_tensor.ptr(), OUTPUT_PEOPLE, OUTPUT_SAMPLES, confidence);
//    g_cpu_tensor.copyFrom(g_gpu_tensor);
//    const std::vector<float> &data = g_cpu_tensor.getData();
//
//    results.clear();
//    for (int i = 0; i < 8400; ++i) {
//        if (data[i * 56 + 4] > confidence) {
//            YoloResult pose_result = extractYoloResult(data, i, 56);
//
//            // Keypoints extraction
//            for (int j = 0; j < 17; ++j) {
//                YoloPoint keypoint{
//                        static_cast<int>(data[i * 56 + 5 + j * 3]),
//                        static_cast<int>(data[i * 56 + 5 + j * 3 + 1]),
//                        data[i * 56 + 5 + j * 3 + 2]
//                };
//                pose_result.keypoints.push_back(keypoint);
//            }
//            results.push_back(pose_result);
//        }
//    }
//}



void postprocess(const Tensor<float> &input, std::vector<YoloResult> &output, float confidence) {

    // Postprocess the object detection results
    sctYolov8ObjectPostProcessing(input.ptr(), g_output_temp.ptr(), OUTPUT_OBJECTS, OUTPUT_SAMPLES, confidence);

    std::vector<float> cpu_results;
    g_output_temp.copyTo(cpu_results);

    output.clear();
    for (int i = 0; i < 8400; ++i) {
        if (cpu_results[i * 84 + 4] > confidence) {
            output.push_back(extractYoloResult(cpu_results, i, 84));
        }
    }
}
