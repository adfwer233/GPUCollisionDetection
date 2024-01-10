#include "algorithm"
#include "collision_detection/gpu/gpu_collision_detection.hpp"
#include "cub/cub.cuh"
#include "cuda_runtime.h"
#include "gpu_sweep_and_prune_collision_detection_kernel.cuh"
#include "thrust/device_vector.h"

/**
 * @brief GPU version of sweep and prune algorithm
 * @param solid_ref_vector
 * @param time_step
 */
void GPUSweepAndPruneCollisionDetection::collision_detection(
    std::vector<std::reference_wrapper<Solid>> &solid_ref_vector,
    float time_step) {
  /*
   * Step1: construct solid info
   */

  std::vector<SolidInfoGPU> solid_infos;

  // number of solids

  size_t n = solid_ref_vector.size();

  for (size_t i = 0; i < n; i++) {
    auto box = solid_ref_vector[i].get().getBoundingBox();
    solid_infos.emplace_back(i, box);
  }

  /*
   * Step2: Copy info array to gpu memory and sort on x-axis (increase by min_x)
   */

  SolidInfoGPU *dev_solid_info, *dev_solid_info_sorted;
  cudaMalloc(&dev_solid_info, n * sizeof(decltype(solid_infos)::value_type));
  cudaMalloc(&dev_solid_info_sorted,
             n * sizeof(decltype(solid_infos)::value_type));
  cudaMemcpy(dev_solid_info, solid_infos.data(), n * sizeof(SolidInfoGPU),
             cudaMemcpyHostToDevice);

  struct SolidInfoDecomposerT {
    __host__ __device__ cuda::std::tuple<float &> operator()(
        SolidInfoGPU &key) {
      return {key.box.min_x};
    }
  };

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                 dev_solid_info, dev_solid_info_sorted, n,
                                 SolidInfoDecomposerT{});

  /*
   * Step3: get possible collision upper bound via binary search
   */
  size_t *possible_collision_upper_bound = nullptr;
  cudaMalloc(&possible_collision_upper_bound, n * sizeof(size_t));

  find_possible_collision_upper_bound_kernel<<<256, 64>>>(
      dev_solid_info_sorted, possible_collision_upper_bound, n);

  /*
   * step4: find all AABB collision pairs
   */

  thrust::device_vector<thrust::device_vector<size_t>> aabb_collision_result(n);

  find_aabb_collision_pairs_kernel<<<256, 64>>>(
      aabb_collision_result, possible_collision_upper_bound, n);

  /*
   * Step5: collision detection between balls
   */
  thrust::device_vector<std::tuple<size_t, size_t>> gpu_collision_result;

  ball_collision_detection_kernel<<<256, 64>>>(
      aabb_collision_result, gpu_collision_result, dev_solid_info_sorted, n);
}