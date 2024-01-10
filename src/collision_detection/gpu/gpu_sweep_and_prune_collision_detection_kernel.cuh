#include "cuda_runtime.h"

#include "collision_detection/gpu/gpu_collision_detection.hpp"

#include "thrust/device_vector.h"

/**
 * @brief CUDA kernel to find possible collision index
 * @param solid_info
 * @param res
 * @param n
 */
__global__ void find_possible_collision_upper_bound_kernel(SolidInfoGPU* solid_info, size_t *res, int n) {

}

__global__ void find_aabb_collision_pairs_kernel(thrust::device_vector<thrust::device_vector<size_t>> &result, size_t* index_upper_bound, int n) {

}

__global__ void ball_collision_detection_kernel(thrust::device_vector<thrust::device_vector<size_t>> &possible_pairs, thrust::device_vector<std::tuple<size_t, size_t>> &result, SolidInfoGPU* device_info, int n) {

}