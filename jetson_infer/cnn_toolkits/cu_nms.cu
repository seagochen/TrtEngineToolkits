//
// Created by Orlando Chen on 8/23/24.
//

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

// CUDA kernel to perform Non-Maximum Suppression (NMS)
__global__ void nms_kernel(const float* bboxes, const float* scores, int* keep_indices, int* keep_count, int num_boxes, float iou_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    const float x1 = bboxes[idx * 4 + 0];
    const float y1 = bboxes[idx * 4 + 1];
    const float x2 = bboxes[idx * 4 + 2];
    const float y2 = bboxes[idx * 4 + 3];
    const float score = scores[idx];
    const float area = (x2 - x1 + 1) * (y2 - y1 + 1);

    for (int i = 0; i < num_boxes; ++i) {
        if (i == idx || keep_indices[i] == 0) continue;

        const float xx1 = max(x1, bboxes[i * 4 + 0]);
        const float yy1 = max(y1, bboxes[i * 4 + 1]);
        const float xx2 = min(x2, bboxes[i * 4 + 2]);
        const float yy2 = min(y2, bboxes[i * 4 + 3]);

        const float w = max(0.0f, xx2 - xx1 + 1);
        const float h = max(0.0f, yy2 - yy1 + 1);
        const float inter = w * h;
        const float ovr = inter / (area + (bboxes[i * 4 + 2] - bboxes[i * 4 + 0] + 1) * (bboxes[i * 4 + 3] - bboxes[i * 4 + 1] + 1) - inter);

        if (ovr > iou_threshold && score <= scores[i]) {
            keep_indices[idx] = 0;
            return;
        }
    }

    int old = atomicAdd(keep_count, 1);
    keep_indices[old] = idx;
}

// Host function to launch NMS CUDA kernel
void nms_cuda(const std::vector<float>& h_bboxes, const std::vector<float>& h_scores, float iou_threshold, std::vector<int>& keep_indices) {
    int num_boxes = h_scores.size();

    float* d_bboxes;
    float* d_scores;
    int* d_keep_indices;
    int* d_keep_count;

    cudaMalloc(&d_bboxes, h_bboxes.size() * sizeof(float));
    cudaMalloc(&d_scores, h_scores.size() * sizeof(float));
    cudaMalloc(&d_keep_indices, num_boxes * sizeof(int));
    cudaMalloc(&d_keep_count, sizeof(int));

    cudaMemcpy(d_bboxes, h_bboxes.data(), h_bboxes.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, h_scores.data(), h_scores.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_keep_indices, 1, num_boxes * sizeof(int));
    cudaMemset(d_keep_count, 0, sizeof(int));

    int blockSize = 256;
    int gridSize = (num_boxes + blockSize - 1) / blockSize;

    nms_kernel<<<gridSize, blockSize>>>(d_bboxes, d_scores, d_keep_indices, d_keep_count, num_boxes, iou_threshold);

    cudaDeviceSynchronize();

    int h_keep_count;
    cudaMemcpy(&h_keep_count, d_keep_count, sizeof(int), cudaMemcpyDeviceToHost);

    keep_indices.resize(h_keep_count);
    cudaMemcpy(keep_indices.data(), d_keep_indices, h_keep_count * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_bboxes);
    cudaFree(d_scores);
    cudaFree(d_keep_indices);
    cudaFree(d_keep_count);
}

int main() {
    std::vector<float> bboxes = {
            50, 50, 100, 100,
            60, 60, 110, 110,
            55, 55, 105, 105,
            100, 100, 150, 150,
            150, 150, 200, 200
    };
    std::vector<float> scores = {0.9, 0.8, 0.85, 0.7, 0.6};
    float iou_threshold = 0.5;

    std::vector<int> keep_indices;
    nms_cuda(bboxes, scores, iou_threshold, keep_indices);

    for (int i : keep_indices) {
        std::cout << "Keep index: " << i << std::endl;
    }

    return 0;
}
