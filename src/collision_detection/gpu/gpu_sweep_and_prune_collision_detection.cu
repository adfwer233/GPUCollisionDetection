#include "algorithm"
#include "collision_detection/gpu/gpu_collision_detection.hpp"
#include "cub/cub.cuh"
#include "cuda/std/tuple"
#include "cuda_runtime.h"
#include "format"
#include "gpu_sweep_and_prune_collision_detection_kernel.cuh"

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

struct SolidInfoDecomposerT {
    __host__ __device__ ::cuda::std::tuple<float &> operator()(SolidInfoGPU &key) const {
        return {key.min_x};
    }
};

GPUSweepAndPruneCollisionDetection::GPUSweepAndPruneCollisionDetection(int n) {
    cudaMalloc(&dev_solid_info, n * sizeof(SolidInfoGPU));
    cudaMalloc(&dev_solid_info_sorted, n * sizeof(SolidInfoGPU));
    cudaMalloc(&possible_collision_upper_bound, n * sizeof(size_t));
    cudaMalloc(&gpu_collision_result, n * MAX_COLLISION_SOLID_COUNT * sizeof(size_t));
    cudaMalloc(&aabb_collision_result, n * MAX_COLLISION_SOLID_COUNT * sizeof(size_t));

    host_gpu_collision_result = new size_t[n * MAX_COLLISION_SOLID_COUNT];
}

GPUSweepAndPruneCollisionDetection::~GPUSweepAndPruneCollisionDetection() {
    cudaFree(dev_solid_info);
    cudaFree(dev_solid_info_sorted);
    cudaFree(possible_collision_upper_bound);
    cudaFree(gpu_collision_result);
    cudaFree(aabb_collision_result);

    delete host_gpu_collision_result;
}

/**
 * @brief GPU version of sweep and prune algorithm
 * @param solid_ref_vector
 * @param time_step
 */
void GPUSweepAndPruneCollisionDetection::collision_detection(
    std::vector<std::reference_wrapper<Solid>> &solid_ref_vector, float time_step) {
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

    cudaMemcpy(dev_solid_info, solid_infos.data(), n * sizeof(SolidInfoGPU), cudaMemcpyHostToDevice);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, dev_solid_info, dev_solid_info_sorted, n,
                                   SolidInfoDecomposerT{});

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, dev_solid_info, dev_solid_info_sorted, n,
                                   SolidInfoDecomposerT{});

    cudaFree(d_temp_storage);

    /*
     * Step3: get possible collision upper bound via binary search
     */

    const int thread_per_block = 512;

    find_possible_collision_upper_bound_kernel<<<int(n / (thread_per_block * ITEMS_PER_THREAD)) + 1,
                                                 thread_per_block>>>(dev_solid_info_sorted,
                                                                     possible_collision_upper_bound, n);

    /*
     * step4: find all AABB collision pairs
     */

    find_aabb_collision_pairs_kernel<<<int(n / (thread_per_block * ITEMS_PER_THREAD)) + 1, thread_per_block>>>(
        dev_solid_info_sorted, possible_collision_upper_bound, aabb_collision_result, n);

    /*
     * Step5: collision detection between balls
     */

    ball_collision_detection_kernel<<<int(n / (thread_per_block * ITEMS_PER_THREAD)) + 1, thread_per_block>>>(
        dev_solid_info, aabb_collision_result, gpu_collision_result, n);

    cudaMemcpy(host_gpu_collision_result, gpu_collision_result, n * MAX_COLLISION_SOLID_COUNT * sizeof(size_t),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < MAX_COLLISION_SOLID_COUNT; j++) {
            if (host_gpu_collision_result[i * MAX_COLLISION_SOLID_COUNT + j] == i)
                break;
            size_t col_id = host_gpu_collision_result[i * MAX_COLLISION_SOLID_COUNT + j];

            auto &solid_i = solid_ref_vector[i].get();
            auto &solid_j = solid_ref_vector[col_id].get();

            if (true) {
                //              assert(false);
                //              std::cout << std::format("({}, {})", i, col_id) <<
                //              std::endl;
                glm::vec3 previous_i_center = solid_i.center - time_step * solid_i.velocity;
                glm::vec3 previous_j_center = solid_j.center - time_step * solid_j.velocity;

                Ball &ball_i = static_cast<Ball &>(solid_i);
                Ball &ball_j = static_cast<Ball &>(solid_j);

                float previous_distance =
                    glm::length(previous_i_center - previous_j_center) - ball_i.radius - ball_j.radius;
                float relative_speed = glm::length(solid_i.velocity - solid_j.velocity);

                float collision_time = previous_distance / relative_speed;

                if (collision_time < time_step) {
                    auto collision_displacement_i = -solid_i.velocity * (time_step - collision_time) +
                                                    solid_j.velocity * (time_step - collision_time);
                    auto collision_displacement_j = -solid_j.velocity * (time_step - collision_time) +
                                                    solid_i.velocity * (time_step - collision_time);

                    if (solid_i.mesh_ref.has_value() and solid_j.mesh_ref.has_value()) {
                        solid_i.mesh_ref->get().transform =
                            glm::translate(solid_i.mesh_ref->get().transform, collision_displacement_i);
                        solid_j.mesh_ref->get().transform =
                            glm::translate(solid_j.mesh_ref->get().transform, collision_displacement_j);
                    }

                    solid_i.center += collision_displacement_i;
                    solid_j.center += collision_displacement_j;
                }
                std::swap(solid_i.velocity, solid_j.velocity);
            }
        }
    }
}