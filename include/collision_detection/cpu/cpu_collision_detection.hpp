#pragma once

#include "vector"

#include "common/solid/solid.hpp"

/**
 * @brief Class implementing naive collision detection algorithm
 */
class CPUNaiveCollisionDetection {
public:
    static void collision_detection (std::vector<std::reference_wrapper<Solid>> &solid_ref_vector);
};

/**
 * @brief Class implementing CPU version of Sweep and Prune algorithm
 */
class CPUSweepAndPruneCollisionDetection {
private:
    struct SolidInfo {
        size_t id;
        AxisAlignedBoundingBox box;
    };
public:
    static void collision_detection (std::vector<std::reference_wrapper<Solid>> &solid_ref_vector, float time_step);
};