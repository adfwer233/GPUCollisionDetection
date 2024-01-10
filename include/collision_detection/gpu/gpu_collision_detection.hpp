#pragma once

#include "vector"

#include "common/solid/solid.hpp"

struct SolidInfoGPU {
    size_t id;
    AxisAlignedBoundingBox box;
};

/*
 * @brief Class implementing GPU SaP Algorithm
 */
class GPUSweepAndPruneCollisionDetection {
public:
    static void collision_detection (std::vector<std::reference_wrapper<Solid>> &solid_ref_vector, float time_step);
};