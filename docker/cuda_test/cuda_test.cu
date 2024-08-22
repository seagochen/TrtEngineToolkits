#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel function to square elements of an array
__global__ void square(float *d_out, const float *d_in, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

int main() {
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // Generate input array on the host
    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = static_cast<float>(i);
    }

    // Declare GPU memory pointers
    float *d_in;
    float *d_out;

    // Allocate GPU memory
    cudaMalloc((void **)&d_in, ARRAY_BYTES);
    cudaMalloc((void **)&d_out, ARRAY_BYTES);

    // Transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 16;
    int gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    square<<<gridSize, blockSize>>>(d_out, d_in, ARRAY_SIZE);

    // Synchronize the device
    cudaDeviceSynchronize();

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy back the result array to the CPU
    float h_out[ARRAY_SIZE];
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Input Array: \n";
    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << h_in[i] << " ";
    }
    std::cout << "\nOutput Array (Squared): \n";
    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

