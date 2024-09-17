//
// Created by orlando on 9/20/24.
//

#include "InferWrapper.h"

#include <simple_cuda_toolkits/vision/colorspace.h>
#include <simple_cuda_toolkits/vision/normalization.h>
#include <simple_cuda_toolkits/tsutils/permute_3D.h>
#include <simple_cuda_toolkits/bbox/bbox.h>
#include <simple_cuda_toolkits/bbox/find_max.h>
#include <simple_cuda_toolkits/bbox/suppress.h>
#include <simple_cuda_toolkits/bbox/nms.h>
#include <simple_cuda_toolkits/matrix/matrix.h>


// Custom deleter for ICudaEngine
auto engineDeleter = [](nvinfer1::ICudaEngine* engine) {
    if (engine) {
        engine->destroy();
    }
};

// Custom deleter for IExecutionContext
auto contextDeleter = [](nvinfer1::IExecutionContext* context) {
    if (context) {
        context->destroy();
    }
};


InferWrapper::InferWrapper(const std::string &engine_path,
            const std::map<std::string, std::string> &names,
            const nvinfer1::Dims4 &dims0, 
            const nvinfer1::Dims3 &dims1,
            const int boxes):
        engine(nullptr, engineDeleter), context(nullptr, contextDeleter) {
            
    // Load the TensorRT engine from the serialized engine file
    engine = loadEngineFromFile(engine_path);
    if (!engine) {
        throw std::runtime_error("Failed to load engine from file.");
    }

    // Create a context for executing the engine
    context = createExecutionContext(engine, names.at("input"), dims0);
    std::cout << "[InferWrapper/Engine] VERBOSE: Context created successfully." << std::endl;

    // Dimensions of input and output tensors
    input_dims = {dims0.d[0], dims0.d[1], dims0.d[2], dims0.d[3]}; // batch, channels, height, width
    output_dims = {dims1.d[0], dims1.d[1], dims1.d[2]}; // batch, features, samples
    this->boxes = boxes;

    // Allocate input and output buffers for CUDA
    std::map<std::string, std::vector<int>> trt_binding_dims;
    trt_binding_dims[names.at("input")] = input_dims;
    trt_binding_dims[names.at("output")] = output_dims;
    trt_buffers = allocateCudaTensors(trt_binding_dims);
    tensor_names = names;
    std::cout << "[InferWrapper/TRTBuffer] VERBOSE: Buffers for TensorRT engine are ready." << std::endl;

    // Allocate input and output buffers for CUDA
    cuda_input_buffers[0] = createZerosTensor<TensorType::FLOAT32>(input_dims[1] * input_dims[2] * input_dims[3]);
    cuda_input_buffers[1] = createZerosTensor<TensorType::FLOAT32>(input_dims[1] * input_dims[2] * input_dims[3]);
    cuda_output_buffers[0] = createZerosTensor<TensorType::FLOAT32>(output_dims[1] * output_dims[2]);
    cuda_output_buffers[1] = createZerosTensor<TensorType::FLOAT32>(output_dims[1] * output_dims[2]);
    std::cout << "[InferWrapper/CUDABuffer] VERBOSE: Temporary buffers for CUDA are ready." << std::endl;
    std::cout << "[InferWrapper/CUDABuffer] VERBOSE: The shape of input temporary buffers is: " 
        << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3];
    std::cout << " (" << cuda_input_buffers.size() << ")" << std::endl;
    std::cout << "[InferWrapper] VERBOSE: The shape of output temporary buffers is: "
        << output_dims[0] << "x" << output_dims[1] << "x" << output_dims[2];
    std::cout << " (" << cuda_output_buffers.size() << ")" << std::endl;

    // Allocate temporary images for OpenCV
    temp_images["resized"] = cv::Mat(input_dims[2], input_dims[3], CV_8UC3);
    temp_images["floated"] = cv::Mat(input_dims[2], input_dims[3], CV_32FC3);
    std::cout << "[InferWrapper/OpenCVBuffer] VERBOSE: Temporary images for OpenCV are ready." << std::endl;

    // Allocate Bitmap for NMS
    sctAllocateNMSBitmap(this->boxes);
    std::cout << "[InferWrapper/NMS] VERBOSE: Bitmap for NMS is ready." << std::endl;

    // Allocate results buffer
    int batch_size = (output_dims[0] > MAX_BATCH_SIZE) ? MAX_BATCH_SIZE : output_dims[0];
    for (int i = 0; i < batch_size; ++i) {
        results.emplace_back(createZerosTensor<TensorType::FLOAT32>(output_dims[1] * output_dims[2]));
    }
    std::cout << "[InferWrapper/Output] VERBOSE: reference batch size has been set to" <<
        results.size() << " for every turn." << std::endl;
}

InferWrapper::~InferWrapper() {
    // Release resources
    trt_buffers.clear();
    cuda_input_buffers.clear();
    cuda_output_buffers.clear();
    temp_images.clear();

    // Destroy the execution context
    context.reset();

    // Destroy the engine
    engine.reset();

    // Release the results buffer
    sctFreeNMSBitmap();

    std::cout << "[InferWrapper/End] VERBOSE: InferWrapper destructor called." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**   
* @brief Preprocess input image for inference
* @param image Input image for inference
*/
void InferWrapper::preprocess(const cv::Mat &image) {
    // Check if the image index is within the batch size
    if (image_idx >= MAX_BATCH_SIZE) {
        throw std::runtime_error("The batch size is exceeded.");
    }

    // Resize the input image and convert to float32
    cv::resize(image, temp_images["resized"], cv::Size(input_dims[3], input_dims[2]));
    temp_images["resized"].convertTo(temp_images["floated"], CV_32FC3);

    // Width, height, and channels
    int width = input_dims[3];
    int height = input_dims[2];
    int channels = input_dims[1];

    // PTR for the input buffers
    auto ptr0 = cuda_input_buffers[0].ptr();
    auto ptr1 = cuda_input_buffers[1].ptr();

    // Copy the image to the input buffer
    cudaMemcpy(ptr0, temp_images["floated"].data, 
            channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Convert the image to RGB and normalize
    sctBGR2RGB(ptr0, ptr1, width, height, channels);
    sctNormalizeData(ptr1, ptr0, width, height, channels);

    // Permute the input tensor from HWC to CHW
    sctPermute3D(ptr0, ptr1, width, height, channels, 2, 0, 1);

    // Copy the input tensor to the TensorRT buffer
    int offset = channels * height * width * image_idx;
    cudaMemcpy(trt_buffers[tensor_names.at("input")].ptr() + offset, ptr1,
            channels * height * width * sizeof(float), cudaMemcpyDeviceToDevice);

    // Increment the image index
    image_idx++;
}

/**
 * @brief Perform inference on the input images
 * @param images Vector of input images for inference
 */
void InferWrapper::preprocess(const std::vector<cv::Mat> &images) {
    for (const auto &image : images) {
        preprocess(image);
    }
}

/**
 * @brief Perform inference on the input images
 */
void InferWrapper::infer(float cls_threshold, float nms_threshold, float alpha, float beta) {
    // Check if the image index is within the batch size
    if (image_idx == 0) {
        throw std::runtime_error("No image is preprocessed.");
    }
    if (image_idx > results.size()) {
        throw std::runtime_error("The batch size is exceeded.");
    }

    // PTR for the output buffers 
    auto ptr0 = cuda_output_buffers[0].ptr();
    auto ptr1 = cuda_output_buffers[1].ptr();

    // Features and samples
    int features = output_dims[1];
    int samples = output_dims[2];

    // Execute the inference
    inference(context, trt_buffers[tensor_names.at("input")], trt_buffers[tensor_names.at("output")]);

    // Postprocess the output tensor
    for (int i = 0; i < image_idx; ++i) {
        // Copy the output tensor to the CUDA buffer
        checkCudaError(cudaMemcpy(ptr0, trt_buffers[tensor_names.at("output")].ptr() + i * features * samples,
                features * samples * sizeof(float), cudaMemcpyDeviceToDevice), 
                "CUDA cudaMemcpyDeviceToDevice failed.");

        // Transpose the output tensor
        sctMatrixTranspose(ptr0, ptr1, features, samples);

        // Find the maximum value in the tensor
        sctFindMaxScores(ptr1, ptr0, features, samples);

        // Suppress the bounding boxes
        sctSuppressResults_closed(ptr0, ptr1, cls_threshold, features, samples);

        // Convert XYWH to XYXY
        sctXYWH2XYXY(ptr1, ptr0, features, samples, alpha, beta); // Only convert the top 1024 boxes, and the range is 0-640

        // Perform non-maximum suppression
        sctNMS_V2(ptr0, ptr1, nms_threshold, features, boxes);  // Only detect the top 1024 boxes

        // Copy the output tensor to the result buffer
        cudaMemcpy(results[i].ptr(), ptr1, features * samples * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Reset the image index
    image_idx = 0;
}

/**
 * @brief Get the available slots count for storing preprocessed images
 * @return Number of available slots
 */
int InferWrapper::getAvailableSlot() const {
    return results.size() - image_idx;
}


void InferWrapper::getResults(int idx, void(*callback)(float*, size_t)) {

    if (idx >= results.size()) {
        throw std::runtime_error("Invalid index for getting results.");
    }

    // Copy the results to CPU
    
    results[idx].copyTo(raw_output);

    // Callback function for processing the results
    callback(raw_output.data(), raw_output.size());
}