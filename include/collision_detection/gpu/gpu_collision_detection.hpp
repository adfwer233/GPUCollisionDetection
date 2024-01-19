#pragma once

#include "common/solid/solid.hpp"
#include "vector"

/**
 * @brief Struct to use in GPU collision detection algorithm
 */
struct SolidInfoGPU {
    size_t id;
    float min_x, max_x, min_y, max_y, min_z, max_z;
};

/**
 * @brief Class implementing GPU SaP Algorithm
 *
 * This class follows RAII design pattern.
 */
class GPUSweepAndPruneCollisionDetection {
  private:
    SolidInfoGPU *dev_solid_info{}, *dev_solid_info_sorted{};
    size_t *possible_collision_upper_bound{}, *aabb_collision_result{}, *gpu_collision_result{};
    size_t *host_gpu_collision_result{};

  public:
    /**
     * @brief create instance and alloc gpu memory
     * @param n
     */
    explicit GPUSweepAndPruneCollisionDetection(int n);

    /**
     * @brief deconstructor, gpu memory will be free
     */
    ~GPUSweepAndPruneCollisionDetection();

    /**
     * @brief GPU SaP Algorithm
     * @param solid_ref_vector vector of ( references of solids )
     * @param time_step time step of simulator
     */
    void collision_detection(std::vector<std::reference_wrapper<Solid>> &solid_ref_vector, float time_step);
};