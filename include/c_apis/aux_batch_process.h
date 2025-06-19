#ifndef AUX_BATCH_PROCESS_H
#define AUX_BATCH_PROCESS_H

#include <vector>
#include "c_apis/c_dstruct.h"
#include "serverlet/models/infer_model_multi.h"

// Struct to hold results and images from the pose detection stage
struct PoseBatchOutput {
    // Each inner vector holds C_Extended_Person_Feats for one image
    std::vector<std::vector<C_Extended_Person_Feats>> detections_per_image;
    // Each cv::Mat corresponds to an image that was processed, resized to model input dimensions (e.g., 640x640)
    std::vector<cv::Mat> processed_images; 
    bool success; // Indicates if the overall processing was successful
};

// Function signature for the updated batch processing function
PoseBatchOutput process_batch_images_by_pose_engine(
    std::vector<cv::Mat>& images, // Input images (will be consumed/popped)
    const std::unique_ptr<InferModelBaseMulti>& pose_model,
    const std::map<std::string, std::any>& pose_pp_params,
    const int pose_max_batch_size
);

#endif // AUX_BATCH_PROCESS_H