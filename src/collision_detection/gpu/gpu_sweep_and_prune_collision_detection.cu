#include "algorithm"
#include "collision_detection/gpu/gpu_collision_detection.hpp"
#include "cub/cub.cuh"
#include "cuda_runtime.h"
#include "gpu_sweep_and_prune_collision_detection_kernel.cuh"

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

struct SolidInfoDecomposerT {
    __host__ __device__ ::cuda::std::tuple<float &, size_t &>
    operator()(SolidInfoGPU &key) const
    {
        return {key.min_x, key.id};
    }
};

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
    solid_infos.push_back({i, box.min_x, box.max_x, box.min_y, box.max_y, box.min_z, box.max_z});
  }

  /*
   * Step2: Copy info array to gpu memory and sort on x-axis (increase by min_x)
   */

  SolidInfoGPU *dev_solid_info;
  thrust::device_vector<SolidInfoGPU> dev_solid_info_sorted(n);

  cudaMalloc(&dev_solid_info, n * sizeof(decltype(solid_infos)::value_type));
  cudaMemcpy(dev_solid_info, solid_infos.data(), n * sizeof(SolidInfoGPU),
             cudaMemcpyHostToDevice);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  SolidInfoGPU* raw_dev_solid_info_sorted = thrust::raw_pointer_cast(dev_solid_info_sorted.data());

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                 dev_solid_info, raw_dev_solid_info_sorted, n,
                                 SolidInfoDecomposerT{});

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, dev_solid_info, raw_dev_solid_info_sorted, n, SolidInfoDecomposerT{});

  /*
   * Step3: get possible collision upper bound via binary search
   */
  thrust::device_vector<size_t> possible_collision_upper_bound(n);
  size_t *raw_possible_collision_upper_bound = thrust::raw_pointer_cast(possible_collision_upper_bound.data());

  find_possible_collision_upper_bound_kernel<<<256, 64>>>(
      raw_dev_solid_info_sorted, raw_possible_collision_upper_bound, n);

  /*
   * step4: find all AABB collision pairs
   */

  thrust::device_vector<size_t> aabb_collision_result(n * MAX_COLLISION_SOLID_COUNT);
  size_t *raw_aabb_collision_result = thrust::raw_pointer_cast(aabb_collision_result.data());

  find_aabb_collision_pairs_kernel<<<256, 64>>>(
      raw_dev_solid_info_sorted,
      raw_possible_collision_upper_bound,
      raw_aabb_collision_result, n);

  /*
   * Step5: collision detection between balls
   */
  thrust::device_vector<size_t> gpu_collision_result(n * MAX_COLLISION_SOLID_COUNT);
  size_t *raw_gpu_collision_result = thrust::raw_pointer_cast(gpu_collision_result.data());

  ball_collision_detection_kernel<<<256, 64>>>(raw_dev_solid_info_sorted, raw_aabb_collision_result, raw_gpu_collision_result, n);

  thrust::host_vector<size_t> host_gpu_collision_result(n * MAX_COLLISION_SOLID_COUNT);

  thrust::copy(gpu_collision_result.begin(), gpu_collision_result.end(), host_gpu_collision_result.begin());

//  thrust::host_vector<SolidInfoGPU> host_solid_info(n);
//  thrust::copy(solid_infos.begin(), solid_infos.end(), host_solid_info.begin());
//
//  thrust::host_vector<size_t> host_possible(n);
//  thrust::copy(possible_collision_upper_bound.begin(), possible_collision_upper_bound.end(), host_possible.begin());
//
//    thrust::host_vector<size_t> host_aabb(n * MAX_COLLISION_SOLID_COUNT);
//    thrust::copy(aabb_collision_result.begin(), aabb_collision_result.end(), host_aabb.begin());


//  for (int i = 0; i < 50; i++) {
//      std::cout << host_possible[i] << ' ';
//  }
//  std::cout << std::endl;
//
//  for (int i = 0; i < 50; i++) {
//      std::cout << host_aabb[i] << ' ';
//  }
//  std::cout << std::endl;
//
//    for (int i = 0; i < 50; i++) {
//        std::cout << host_gpu_collision_result[i] << ' ';
//    }
//    std::cout << std::endl;
//
//  std::cout << host_gpu_collision_result.size() << std::endl;

  for (int i = 0; i < n; i++) {
      for (int j = 0; j < MAX_COLLISION_SOLID_COUNT; j++) {
          if (host_gpu_collision_result[i * MAX_COLLISION_SOLID_COUNT + j] == i) break;
          size_t col_id = host_gpu_collision_result[i * MAX_COLLISION_SOLID_COUNT + j];

          auto &solid_i = solid_ref_vector[i].get();
          auto &solid_j = solid_ref_vector[col_id].get();

          if (true) {
//              assert(false);
              std::cout << std::format("({}, {})", i, col_id) << std::endl;
              glm::vec3 previous_i_center =
                      solid_i.center - time_step * solid_i.velocity;
              glm::vec3 previous_j_center =
                      solid_j.center - time_step * solid_j.velocity;

              Ball &ball_i = static_cast<Ball &>(solid_i);
              Ball &ball_j = static_cast<Ball &>(solid_j);

              float previous_distance =
                      glm::length(previous_i_center - previous_j_center) - ball_i.radius -
                      ball_j.radius;
              float relative_speed = glm::length(solid_i.velocity - solid_j.velocity);

              float collision_time = previous_distance / relative_speed;

              if (collision_time < time_step) {
                  auto collision_displacement_i =
                          -solid_i.velocity * (time_step - collision_time) +
                          solid_j.velocity * (time_step - collision_time);
                  auto collision_displacement_j =
                          -solid_j.velocity * (time_step - collision_time) +
                          solid_i.velocity * (time_step - collision_time);

                  if (solid_i.mesh_ref.has_value() and solid_j.mesh_ref.has_value()) {
                      solid_i.mesh_ref->get().transform = glm::translate(
                              solid_i.mesh_ref->get().transform, collision_displacement_i);
                      solid_j.mesh_ref->get().transform = glm::translate(
                              solid_j.mesh_ref->get().transform, collision_displacement_j);
                  }

                  solid_i.center += collision_displacement_i;
                  solid_j.center += collision_displacement_j;
              } else {
                  assert(false);
              }
              std::swap(solid_i.velocity, solid_j.velocity);
          }
      }
  }
}