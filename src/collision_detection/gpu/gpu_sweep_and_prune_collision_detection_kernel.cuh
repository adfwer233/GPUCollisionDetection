#include "collision_detection/gpu/gpu_collision_detection.hpp"
#include "cuda_runtime.h"
#include "thrust/binary_search.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include <span>

constexpr int ITEMS_PER_THREAD = 16;
constexpr int MAX_COLLISION_SOLID_COUNT = 10;

template <class ForwardIt, class T, class PredT>
__device__ ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T &value, PredT pred) {
    ForwardIt it;
    typename std::iterator_traits<ForwardIt>::difference_type count, step;
    count = last - first;

    while (count > 0) {
        it = first;
        step = count / 2;
        it = it + step;

        if (!pred(value, *it)) {
            first = ++it;
            count -= step + 1;
        } else
            count = step;
    }

    return first;
}

/**
 * @brief CUDA kernel to find possible collision index
 * @param solid_info
 * @param res
 * @param n
 */
__global__ void find_possible_collision_upper_bound_kernel(SolidInfoGPU *solid_info, size_t *res, int n) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        size_t id = tid * ITEMS_PER_THREAD + i;

        if (id < n) {
            //            auto ptr_info = thrust::raw_pointer_cast(solid_info.data() +
            //            id);
            auto it = upper_bound(solid_info, solid_info + n, solid_info[id].max_x,
                                  [](float x, SolidInfoGPU &y) { return x < y.min_x; });
            res[id] = it - solid_info;
        }
    }
}

__global__ void find_aabb_collision_pairs_kernel(SolidInfoGPU *solid_info, size_t *possible_upper_bound, size_t *result,
                                                 int n) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = 0; i < ITEMS_PER_THREAD; i++) {
        size_t id = tid * ITEMS_PER_THREAD + i;

        if (id >= n)
            break;

        int collision_count = 0;

        size_t upper_bound = possible_upper_bound[id];

        SolidInfoGPU info_i = solid_info[id];

        for (size_t j = id + 1; j < upper_bound; j++) {
            if (collision_count >= MAX_COLLISION_SOLID_COUNT)
                break;

            bool overlap_flag = true;
            if (cub::max(info_i.min_x, solid_info[j].min_x) > cub::min(info_i.max_x, solid_info[j].max_x))
                overlap_flag = false;
            if (cub::max(info_i.min_y, solid_info[j].min_y) > cub::min(info_i.max_y, solid_info[j].max_y))
                overlap_flag = false;
            if (cub::max(info_i.min_z, solid_info[j].min_z) > cub::min(info_i.max_z, solid_info[j].max_z))
                overlap_flag = false;

            if (overlap_flag) {
                result[info_i.id * MAX_COLLISION_SOLID_COUNT + collision_count] = solid_info[j].id;
                collision_count++;
            }
        }

        for (size_t j = collision_count; j < MAX_COLLISION_SOLID_COUNT; j++) {
            result[info_i.id * MAX_COLLISION_SOLID_COUNT + j] = info_i.id;
        }
    }
}

__global__ void ball_collision_detection_kernel(SolidInfoGPU *solid_info, const size_t *aabb_collision, size_t *result,
                                                int n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = 0; i < ITEMS_PER_THREAD; i++) {
        size_t id = tid * ITEMS_PER_THREAD + i;

        if (id >= n)
            break;

        SolidInfoGPU info_i = solid_info[id];

        float i_center_x = 0.5f * (info_i.min_x + info_i.max_x);
        float i_center_y = 0.5f * (info_i.min_y + info_i.max_y);
        float i_center_z = 0.5f * (info_i.min_z + info_i.max_z);
        float i_radius = info_i.max_x - i_center_x;

        int collision_count = 0;

        for (int j = 0; j < MAX_COLLISION_SOLID_COUNT; j++) {
            if (aabb_collision[id * MAX_COLLISION_SOLID_COUNT + j] == id)
                break;
            size_t id_j = aabb_collision[id * MAX_COLLISION_SOLID_COUNT + j];
            float j_center_x = 0.5f * (solid_info[id_j].min_x + solid_info[id_j].max_x);
            float j_center_y = 0.5f * (solid_info[id_j].min_y + solid_info[id_j].max_y);
            float j_center_z = 0.5f * (solid_info[id_j].min_z + solid_info[id_j].max_z);
            float j_radius = solid_info[id_j].max_x - j_center_x;

            if ((i_radius + j_radius) * (i_radius + j_radius) >
                (i_center_x - j_center_x) * (i_center_x - j_center_x) +
                    (i_center_y - j_center_y) * (i_center_y - j_center_y) +
                    (i_center_z - j_center_z) * (i_center_z - j_center_z)) {
                result[id * MAX_COLLISION_SOLID_COUNT + collision_count] = id_j;
                collision_count++;
            }
        }

        for (int j = collision_count; j < MAX_COLLISION_SOLID_COUNT; j++) {
            result[id * MAX_COLLISION_SOLID_COUNT + j] = id;
        }
    }
}