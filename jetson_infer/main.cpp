#include <opencv2/opencv.hpp>
#include <map>
#include "common/engine_loader.h"
#include "common/tensor.hpp"
#include "common/tensor_utils.hpp"

// 从 TensorRT 引擎中加载张量
std::map<std::string, CudaTensor<float>> loadTensorsFromModel(nvinfer1::ICudaEngine* engine) {

    // 创建输入输出缓冲区
    std::map<std::string, CudaTensor<float>> buffers;

    // 获取模型的所有张量名字
    auto tensor_names = getTensorNamesFromModel(engine);

    for (const auto& name : tensor_names) {
        // 获取每个张量的大小
        TensorDimensions dims = getTensorDimsByName(engine, name, tensor_type::FLOAT32);

        // 创建一个 CudaTensor
        CudaTensor<float> tensor(dims);

        // 保存 Tensor 到缓冲区
        buffers[name] = std::move(tensor);

        // 输出一些信息
        std::cout << "Allocated buffer for Tensor: " << name
                  << " with size: " << buffers[name].getMemSize() << " bytes." << std::endl;
    }

    // 返回给上级调用者
    return buffers;
}


int main() {
    // 使用智能指针加载 TensorRT 引擎
    auto engine = loadEngine("./models/yolov8n.engine");
    if (!engine) {
        std::cerr << "Failed to load engine." << std::endl;
        return -1;
    }

    // 使用智能指针创建执行上下文
    auto context = createExecutionContext(engine.get());

    // 获取与模型有关的 Tensors
    auto gpu_tensors = loadTensorsFromModel(engine.get());

    // 打开视频文件
    cv::VideoCapture cap("./res/vehicle_camera_02.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video file." << std::endl;
        return -1;
    }

    // 创建临时数据
    std::vector<float> float_data;
    float_data.resize(640 * 640 * 3);

    std::vector<uint8_t> uint8_data;
    uint8_data.resize(640 * 640 * 3);

    std::vector<float> float_output;
    float_output.resize(640 * 640 * 3);

    std::vector<uint8_t> uint8_output;
    uint8_output.resize(640 * 640 * 3);

    // 视频帧处理
    cv::Mat frame;
    cv::Mat output_frame(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    while (cap.read(frame)) {
        // 调整图像尺寸
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(640, 640));

        // 将数据从unchar ptr转换为vector<uint8_t>
        std::copy(resized_frame.data, resized_frame.data + resized_frame.total() * resized_frame.channels(),
                  uint8_data.begin());

        // 将数据从 uint8_t 转换为 float
        convertUint8ToFloat32(uint8_data, float_data);

        // 将数据复制到 GPU
        gpu_tensors["images"].copyFrom(float_data);

        // 执行推理
        inference(context, gpu_tensors["images"], gpu_tensors["output0"]);

        // 将数据从 GPU 复制到 CPU
        gpu_tensors["images"].copyTo(float_output);

        // 将数据重新拷贝到 uint8_t
        convertFloat32ToUint8(float_output, uint8_output);

        // 将数据拷贝到Mat
        std::copy(uint8_output.begin(), uint8_output.end(), output_frame.data);

        // 显示帧
//        cv::imshow("Frame", output_frame);
//        if (cv::waitKey(1) == 27) {
//            break;
//        }

        std::cout << "..." << std::endl;
    }

    // 调试和消息输出
    std::cout << "Done" << std::endl;

    return 0;
}
