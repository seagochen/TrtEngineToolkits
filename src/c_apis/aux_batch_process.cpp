#include <vector>
#include <opencv2/opencv.hpp>
#include <any>
#include <map>
#include <cstddef> // For size_t
#include <algorithm> // For std::min

#include "trtengine/c_apis/aux_batch_process.h" // Includes the new PoseBatchOutput struct
#include "trtengine/c_apis/c_dstruct.h"         // C-style structs
#include "trtengine/serverlet/models/infer_model_multi.h" // C++ ModelBase
#include "trtengine/serverlet/models/inference/infer_yolo_v8.hpp" // To access specific YoloPose type for casting (e.g., getMaximumBatchSize())
#include "trtengine/utils/logger.h"


// Helper function: Converts C++ YoloPose structs to C-friendly C_Extended_Person_Feats structs.
// In this stage, class_id is initialized to -1, to be filled by EfficientNet later.
void convert_yolopose_to_c_struct(
    const std::vector<YoloPose>& cpp_poses,
    std::vector<C_Extended_Person_Feats>& c_feats)
{
    c_feats.clear();
    c_feats.reserve(cpp_poses.size()); // Pre-allocate space

    for (const auto& cpp_pose : cpp_poses) {
        C_Extended_Person_Feats c_feat;

        // Copy bounding box information (lx, ly, rx, ry)
        c_feat.box.x1 = cpp_pose.lx;
        c_feat.box.y1 = cpp_pose.ly;
        c_feat.box.x2 = cpp_pose.rx;
        c_feat.box.y2 = cpp_pose.ry;
        c_feat.confidence = cpp_pose.conf;
        c_feat.class_id = -1; // Default value, to be filled by EfficientNet stage

        // Copy keypoint information
        c_feat.num_kps = std::min(cpp_pose.pts.size(), (size_t)17); // Limit to max 17 keypoints
        for (int k = 0; k < c_feat.num_kps; ++k) {
            c_feat.kps[k].x = cpp_pose.pts[k].x;
            c_feat.kps[k].y = cpp_pose.pts[k].y;
            c_feat.kps[k].score = cpp_pose.pts[k].conf; // Ensure this matches your YoloPose::Point struct
        }

        // Fill any remaining keypoints with default values (0.0f)
        for (int k = c_feat.num_kps; k < 17; ++k) {
            c_feat.kps[k] = {0.0f, 0.0f, 0.0f};
        }

        c_feats.push_back(c_feat); // Add to the results list
    }
}


// Updated function signature and implementation
PoseBatchOutput process_batch_images_by_pose_engine(
    std::vector<cv::Mat>& images, // Input images (will be consumed/popped)
    const std::unique_ptr<InferModelBaseMulti>& pose_model,
    const std::map<std::string, std::any>& pose_pp_params,
    const int pose_max_batch_size
)
{
    // Initialize the return struct
    PoseBatchOutput output;
    output.success = false; // Assume failure until proven otherwise

    // Check if pose model is initialized
    if (!pose_model) {
        LOG_ERROR("BatchProcess", "Pose model not initialized. Cannot process images.");
        return output; // Return failure
    }

    // Check if input images vector is valid
    if (images.empty()) {
        LOG_WARNING("BatchProcess", "Input images vector is empty. No images to process.");
        output.success = true; // No images to process, consider it a success for an empty input
        return output;
    }

    // Ensure pose_max_batch_size is a valid positive number
    if (pose_max_batch_size <= 0) {
        LOG_ERROR("BatchProcess", "Invalid pose_max_batch_size: " + std::to_string(pose_max_batch_size) + ". Must be greater than 0.");
        return output; // Return failure
    }

    size_t remaining_images_count = images.size(); // Number of images still to process

    LOG_INFO("BatchProcess", "Starting batch processing with pose engine for " + std::to_string(remaining_images_count) + " images.");

    while (remaining_images_count > 0) {
        // Calculate the number of images to process in the current batch
        size_t current_batch_count = std::min(remaining_images_count, (size_t)pose_max_batch_size);

        // Store resized images for this batch. These will be passed to the next stage.
        std::vector<cv::Mat> current_batch_resized_images;
        
        // Collect and preprocess images for the current batch
        for (size_t i = 0; i < current_batch_count; ++i) {
            // Get and remove the last image from the input vector
            cv::Mat image_to_process = images.back();
            images.pop_back(); 
            
            // YOLOv8 models usually expect 640x640 input.
            // Resize the image here before preprocessing and storing.
            cv::Mat resized_image_for_model;
            cv::resize(image_to_process, resized_image_for_model, cv::Size(640, 640));

            // Preprocess the image and push it to the model's input buffer
            pose_model->preprocess(resized_image_for_model, i);
            
            // Store a copy of the *resized* image for later stages (EfficientNet)
            current_batch_resized_images.push_back(resized_image_for_model.clone()); 
        }
        
        // Update the count of remaining images in the input vector
        remaining_images_count = images.size(); 

        LOG_VERBOSE_TOPIC("BatchProcess", "Pose", "Preprocessed " + std::to_string(current_batch_count) + " images for current batch.");

        // Execute model inference
        if (!pose_model->inference()) {
            LOG_ERROR("BatchProcess", "Pose model inference failed for current batch.");
            // output.detections_per_image remains empty, output.processed_images remains empty or partial
            return output; // Return failure
        }
        LOG_VERBOSE_TOPIC("BatchProcess", "Pose", "Inference completed for current batch.");

        // Postprocess and collect detection results for each image in the current batch
        // The results are stored in `temp_batch_results_per_image`, which has one element per image.
        std::vector<std::vector<C_Extended_Person_Feats>> temp_batch_results_per_image;
        for (size_t i = 0; i < current_batch_count; ++i) {
            std::any pose_raw_results;
            // `pose_pp_params` should contain `cls` and `iou` for YOLO post-processing
            pose_model->postprocess(i, pose_pp_params, pose_raw_results);

            try {
                // Convert std::any result to std::vector<YoloPose>
                std::vector<YoloPose> cpp_pose_detections = std::any_cast<std::vector<YoloPose>>(pose_raw_results);
                
                // C-interface result container for this specific image
                std::vector<C_Extended_Person_Feats> c_pose_feats_for_image;
                
                // Convert YoloPose to C_Extended_Person_Feats
                convert_yolopose_to_c_struct(cpp_pose_detections, c_pose_feats_for_image);
                
                // Add current image's results to the temporary batch results
                temp_batch_results_per_image.push_back(std::move(c_pose_feats_for_image));

            } catch (const std::bad_any_cast& e) {
                LOG_ERROR("BatchProcess", "Error casting pose results for batch item " + std::to_string(i) + ": " + std::string(e.what()));
                temp_batch_results_per_image.push_back({}); // Push empty results for this item
            } catch (const std::exception& e) {
                LOG_ERROR("BatchProcess", "An unexpected error during pose postprocessing for batch item " + std::to_string(i) + ": " + std::string(e.what()));
                temp_batch_results_per_image.push_back({}); // Push empty results for this item
            }
        }
        
        // Move the results (detections and images) from the current batch to the final output struct
        // Each element of `temp_batch_results_per_image` is a vector of persons for one image.
        // `output.detections_per_image` needs to store these individual image results.
        for (auto& image_person_results : temp_batch_results_per_image) {
            output.detections_per_image.push_back(std::move(image_person_results));
        }
        // Also move the processed images to the output struct
        for (auto& img : current_batch_resized_images) {
            output.processed_images.push_back(std::move(img));
        }

        LOG_VERBOSE_TOPIC("BatchProcess", "Pose", "Postprocess completed for current batch.");
    }

    output.success = true; // Mark as successful after processing all images
    LOG_INFO("BatchProcess", "Finished processing all images by pose engine. Total images processed: " + std::to_string(output.processed_images.size()));
    return output;
}