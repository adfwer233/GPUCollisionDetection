#pragma once

#include "common/solid/solid.hpp"
#include "vector"

struct SolidInfoGPU {
    size_t id;
    float min_x, max_x, min_y, max_y, min_z, max_z;
};

/*
 * @brief Class implementing GPU SaP Algorithm
 */
class GPUSweepAndPruneCollisionDetection {
  private:
    SolidInfoGPU *dev_solid_info{}, *dev_solid_info_sorted{};
    size_t *possible_collision_upper_bound{}, *aabb_collision_result{}, *gpu_collision_result{};
    size_t *host_gpu_collision_result{};

  public:
    explicit GPUSweepAndPruneCollisionDetection(int n);

    ~GPUSweepAndPruneCollisionDetection();
    void collision_detection(std::vector<std::reference_wrapper<Solid>> &solid_ref_vector, float time_step);
};